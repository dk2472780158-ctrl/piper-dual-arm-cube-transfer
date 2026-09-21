#!/usr/bin/env python3
"""cube_rerun_fast 的无硬件验证脚本：不需要机械臂，不碰 CAN / 相机。

用法（在卡片终端里跑，先 `cd` 到本目录）：

  python -B probe_fastlog.py unit                          # 判帧 / 丢帧逻辑自检，不连服务端
  python -B probe_fastlog.py blackhole --variant fast      # A：TCP 黑洞当 sink，证明控制线程不被卡
  python -B probe_fastlog.py blackhole --variant original  # A 对照：原版会被卡住（有守卫，不会真卡死）
  python -B probe_fastlog.py timing --port 9877 --seconds 180 --variant fast   # C：对照计时
  python -B probe_fastlog.py file --variant fast --out /tmp/pf_fast.rrd        # B：落盘供离线检查
  python -B probe_fastlog.py compare --a /tmp/pf_orig.rrd --b /tmp/pf_fast.rrd

blackhole / timing / file 三个模式会 import rerun（本项目实测首次约 10~20 秒），
unit 模式不 import rerun。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# 平台容器是登录 shell 且设了 PYTHONSAFEPATH=1 —— 那样直接 `python probe_fastlog.py`
# 时**脚本自己的目录不会进 sys.path**，同目录的 cube_fastlog 就 import 不到。
# 这里显式补上，跟启动脚本走 PYTHONPATH 那条路互不干扰。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

N_SCALARS = 28
N_ACTIONS = 14
CAMERAS = ("observation.left", "observation.middle", "observation.right")
IMG_SHAPE = (480, 640, 3)  # 真机三路相机就是这个尺寸
PAYLOAD = "flat"           # flat=纯色块（JPEG 压得动，负载小）| noise=噪声图（压不动，约 0.5 MB/帧）


# --------------------------------------------------------------- 合成数据
def make_observation(step: int) -> dict:
    obs = {f"observation.joint_{i}.pos": np.float32(np.sin(0.1 * step + i)) for i in range(N_SCALARS)}
    for k, name in enumerate(CAMERAS):
        if PAYLOAD == "noise":
            # 噪声图 JPEG q95 几乎压不动（约 0.5 MB/帧），用来把"下游卡住"放大到几十步内必现。
            # 真机的复位阶段本来就每步发 3 路未压缩图（约 2.76 MB），这个量级在真实范围之内。
            img = np.random.default_rng(1000 + step).integers(0, 256, IMG_SHAPE, dtype=np.uint8)
        else:
            img = np.zeros(IMG_SHAPE, dtype=np.uint8)
            img[:, :, k] = (step * 3) % 256      # 每步都在变，方便肉眼看出画面在刷新
            img[10:60, 10:60, :] = (step * 7) % 256
        obs[name] = img
    return obs


def make_action(step: int) -> dict:
    return {f"action.joint_{i}.pos": np.float32(np.cos(0.1 * step + i)) for i in range(N_ACTIONS)}


def camera_keys(obs: dict) -> int:
    return sum(1 for k in obs if k in CAMERAS)


def scalar_keys(obs: dict) -> int:
    return sum(1 for k in obs if k.startswith("observation.joint_"))


# --------------------------------------------------------------- 公共零件
def load_hooks():
    """拿到 (init_rerun, 原版 log_rerun_data, 快版 logger)。

    快版通过 cube_fastlog.install() 显式构造，不依赖 sitecustomize 是否生效，
    这样本脚本在装/不装补丁的环境里都能跑。
    """
    import cube_fastlog
    from lerobot.utils.visualization_utils import init_rerun
    import lerobot.utils.visualization_utils as viz

    original = getattr(viz, "_cube_rerun_fast_original", None)
    if original is None:
        original = viz.log_rerun_data  # 没装补丁时，模块里的就是原版
    fast = cube_fastlog.install(original)
    return init_rerun, original, fast


def start_watchdog(call_timeout: float, total_timeout: float):
    """单次调用超过 call_timeout 秒就判定"被下游卡住"，强制结束进程。

    必须有这个守卫：原版 rr.log() 的阻塞是 SDK 行为，可能一直不返回。
    """
    state = {"t": time.perf_counter(), "t0": time.perf_counter()}

    def run():
        while True:
            time.sleep(0.2)
            now = time.perf_counter()
            if now - state["t"] > call_timeout:
                print(
                    f"\n[probe] 单次 log 调用超过 {call_timeout:g} s 未返回 —— "
                    f"判定为被下游卡死，强制中止本次进程（这正是真机上「冻住半秒到一秒」的机制）",
                    flush=True,
                )
                os._exit(3)
            if total_timeout and now - state["t0"] > total_timeout:
                print(f"\n[probe] 总时长超过 {total_timeout:g} s，正常收尾", flush=True)
                os._exit(0)

    import threading

    threading.Thread(target=run, daemon=True).start()
    return state


def stats(samples):
    if not samples:
        return {}
    a = np.asarray(samples, dtype=np.float64)
    return {
        "n": len(a),
        "p50_ms": float(np.percentile(a, 50) * 1e3),
        "p95_ms": float(np.percentile(a, 95) * 1e3),
        "max_ms": float(a.max() * 1e3),
        "总秒": float(a.sum()),
    }


def fmt(name, s):
    if not s:
        return f"{name}: (无样本)"
    return (
        f"{name}: n={s['n']} p50={s['p50_ms']:.2f} ms p95={s['p95_ms']:.2f} ms "
        f"max={s['max_ms']:.2f} ms 合计={s['总秒']:.1f} s"
    )


# --------------------------------------------------------------- unit
def mode_unit(args) -> int:
    import cube_fastlog

    os.environ.setdefault("CUBE_RERUN_MODE", "async")
    os.environ["CUBE_RERUN_IMG_HZ"] = "5"
    os.environ["CUBE_RERUN_QUEUE"] = "3"

    ok = True
    print("== 1) 判帧：标量每步都发、相机按 5 Hz 降频、一律按 JPEG ==")
    forwarded = []

    def stub(**kw):
        forwarded.append(kw)

    logger = cube_fastlog.FastLogger(stub)
    n = 60
    per_call = []
    for i in range(n):
        t0 = time.perf_counter()
        logger(observation=make_observation(i), action=make_action(i))
        per_call.append(time.perf_counter() - t0)
        time.sleep(1 / 30)  # 按真机的 30 Hz 节流；不节流的话 5 Hz 判帧一共只放行一两次
    control_ms = float(np.median(per_call)) * 1e3
    control_max_ms = float(np.max(per_call)) * 1e3
    logger.close(timeout=10)

    scalars_ok = all(scalar_keys(c["observation"]) == N_SCALARS for c in forwarded)
    acts_ok = all(len(c["action"]) == N_ACTIONS for c in forwarded)
    compress_ok = all(c["compress_images"] is True for c in forwarded)
    img_batches = sum(1 for c in forwarded if camera_keys(c["observation"]) == len(CAMERAS))
    print(f"  转发次数={len(forwarded)}（应为 {n}）")
    print(f"  每步都带满 {N_SCALARS} 条关节标量：{scalars_ok}")
    print(f"  每步都带满 {N_ACTIONS} 条动作标量：{acts_ok}")
    print(f"  一律 compress_images=True：{compress_ok}")
    print(f"  带相机图的批次={img_batches}（60 步按 5 Hz 约 10 次）")
    print(f"  控制线程侧中位耗时={control_ms:.3f} ms/步，最大={control_max_ms:.3f} ms（预算 33.3 ms）")
    ok &= scalars_ok and acts_ok and compress_ok and len(forwarded) == n
    ok &= 8 <= img_batches <= 12
    ok &= control_ms < 5.0

    print("\n== 2) 下游卡住时：控制线程不阻塞、队列丢最旧 ==")
    slow_done = []

    def slow(**kw):
        time.sleep(0.25)  # 模拟"下游跟不上，原函数一调就卡 250 ms"
        slow_done.append(1)

    lg2 = cube_fastlog.FastLogger(slow)
    n2 = 30
    t0 = time.perf_counter()
    for i in range(n2):
        lg2(observation=make_observation(i), action=make_action(i))
        time.sleep(1 / 30)  # 模拟 30 Hz 控制循环
    wall = time.perf_counter() - t0
    lg2.close(timeout=0.2)
    print(f"  30 步 @30 Hz 实际墙钟={wall:.2f} s（下游全不卡要 {n2 * 0.25:.1f} s；正常应≈{n2 / 30:.1f} s）")
    print(f"  下游真正处理完的={len(slow_done)} 次，丢弃={lg2.dropped} 次")
    ok &= wall < 1.5 and lg2.dropped > 10
    print(f"\nunit 自检结果：{'通过' if ok else '不通过'}")
    return 0 if ok else 1


# --------------------------------------------------------------- blackhole
def mode_blackhole(args) -> int:
    import socket
    import threading

    init_rerun, original, fast = load_hooks()

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(16)
    held = []

    def acceptor():
        while True:
            try:
                conn, _ = srv.accept()
                held.append(conn)  # 只接受连接，**永不读取** —— 这就是黑洞
            except OSError:
                return

    threading.Thread(target=acceptor, daemon=True).start()
    print(f"[probe] 黑洞 sink 已在 127.0.0.1:{args.port} 就绪（accept 后不读任何字节）"
          f" 负载={args.payload} 步数={args.steps}")

    fn = fast if args.variant == "fast" else original
    init_rerun(session_name="probe_blackhole", ip="127.0.0.1", port=args.port)
    state = start_watchdog(args.call_timeout, args.total_timeout)

    samples = []
    for i in range(args.steps):
        obs, act = make_observation(i), make_action(i)
        state["t"] = time.perf_counter()
        t0 = time.perf_counter()
        fn(observation=obs, action=act, compress_images=True)
        samples.append(time.perf_counter() - t0)
        if i and i % 50 == 0:
            print(f"  已发 {i} 步，最近一步 {samples[-1] * 1e3:.1f} ms", flush=True)

    print(f"\n== 变体 {args.variant} 结果 ==")
    print("  " + fmt("log 调用耗时", stats(samples)))
    if args.variant == "fast":
        print(f"  丢弃={fast.dropped} 相机帧批次={fast.img_batches} 错误={fast.errors}")
        print("  判读：控制线程侧应全程亚毫秒~毫秒级；丢弃计数说明后台已经在丢最旧，")
        print("        下游的黑洞由后台线程承担 —— 这就是真机上不卡的原因。")
    else:
        print("  判读：只要出现 p95/max 到几百毫秒以上，就复现了真机上的「冻住」。")
    sys.stdout.flush()
    os._exit(0)  # 后台还可能卡在 SDK 里，直接退出，别在解释器收尾处等它


# --------------------------------------------------------------- timing
def mode_timing(args) -> int:
    init_rerun, original, fast = load_hooks()

    fn = fast if args.variant == "fast" else original
    init_rerun(session_name="probe_timing", ip="127.0.0.1", port=args.port)
    print(f"[probe] 变体={args.variant} 目标=127.0.0.1:{args.port} "
          f"（请确认这是你自己起的服务端，别连平台那个）")
    state = start_watchdog(args.call_timeout, args.total_timeout)

    steps = args.steps or int(args.seconds * 30)
    samples = []
    t_start = time.perf_counter()
    for i in range(steps):
        obs, act = make_observation(i), make_action(i)
        state["t"] = time.perf_counter()
        t0 = time.perf_counter()
        fn(observation=obs, action=act, compress_images=True)
        samples.append(time.perf_counter() - t0)
        target = t_start + (i + 1) / 30.0
        now = time.perf_counter()
        if target > now:
            time.sleep(target - now)
        if i and i % 300 == 0:
            print(f"  第 {i} 步，最近一步 {samples[-1] * 1e3:.1f} ms，实际 {i / (now - t_start):.1f} Hz", flush=True)

    wall = time.perf_counter() - t_start
    print(f"\n== 变体 {args.variant} 计时（{steps} 步，墙钟 {wall:.1f} s，实际 {steps / wall:.1f} Hz）==")
    print("  " + fmt("log 调用耗时", stats(samples)))
    if args.variant == "fast":
        print(f"  丢弃={fast.dropped} 相机帧批次={fast.img_batches} 错误={fast.errors}")
    first = stats(samples[: len(samples) // 3])
    last = stats(samples[2 * len(samples) // 3:])
    print(f"  前 1/3: p50={first.get('p50_ms', 0):.2f} ms  后 1/3: p50={last.get('p50_ms', 0):.2f} ms"
          "（后段明显变大 = 越跑越卡）")
    sys.stdout.flush()
    os._exit(0)


# --------------------------------------------------------------- file
def mode_file(args) -> int:
    import rerun as rr

    _, original, fast = load_hooks()
    fn = fast if args.variant == "fast" else original

    # 0.31.3 的签名是 init(application_id, *, spawn=...)：第一个参数是位置参数，
    # 老版本那个 session_name= 关键字在这里会 TypeError。
    rr.init(f"probe_file_{args.variant}", spawn=False)
    rr.set_sinks(rr.FileSink(args.out))
    print(f"[probe] 变体={args.variant} 落盘到 {args.out} "
          f"负载={args.payload} 步数={args.steps} 节流={args.hz or '无'}")

    samples = []
    t_start = time.perf_counter()
    for i in range(args.steps):
        obs, act = make_observation(i), make_action(i)
        t0 = time.perf_counter()
        fn(observation=obs, action=act, compress_images=True)
        samples.append(time.perf_counter() - t0)
        if args.hz > 0:  # 按真机的控制频率节流，否则一秒钟灌完看不出"跟不跟得上"
            target = t_start + (i + 1) / args.hz
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
    wall = time.perf_counter() - t_start
    if args.variant == "fast":
        fast.close(timeout=5)
        print(f"  步数={fast.steps} 相机帧批次={fast.img_batches} 丢弃={fast.dropped}")
    print(f"  墙钟={wall:.1f} s（{args.steps / wall:.1f} Hz）  " + fmt("log 调用耗时", stats(samples)))
    size = os.path.getsize(args.out) if os.path.exists(args.out) else 0
    print(f"  文件大小={size} B（{size / max(1, args.steps):.0f} B/步）")
    sys.stdout.flush()
    os._exit(0)


# --------------------------------------------------------------- compare
def mode_compare(args) -> int:
    import re

    def entities(path):
        with open(path, "rb") as fh:
            blob = fh.read()
        out = set()
        for item in re.findall(rb"[ -~]{4,}", blob):
            text = item.decode("ascii", "ignore")
            if ("observation." in text or "action." in text) and " " not in text.strip():
                out.add(text)
        return out

    a, b = entities(args.a), entities(args.b)
    print(f"A({args.a}) 命中实体名 {len(a)} 个")
    print(f"B({args.b}) 命中实体名 {len(b)} 个")
    only_a = sorted(a - b)
    only_b = sorted(b - a)
    print(f"只在 A 里出现：{len(only_a)} 个 {only_a[:8]}")
    print(f"只在 B 里出现：{len(only_b)} 个 {only_b[:8]}")
    same = not only_a and not only_b
    print(f"实体名集合一致：{same}（相机是覆盖式实体，快版只是降频，名字不该变）")
    return 0 if same else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="cube_rerun_fast 无硬件验证脚本")
    ap.add_argument("mode", choices=["unit", "blackhole", "timing", "file", "compare"])
    ap.add_argument("--variant", choices=["fast", "original"], default="fast")
    ap.add_argument("--port", type=int, default=9877)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--out", default="/tmp/probe_fastlog.rrd")
    ap.add_argument("--a", default="/tmp/pf_orig.rrd")
    ap.add_argument("--b", default="/tmp/pf_fast.rrd")
    ap.add_argument("--hz", type=float, default=30.0,
                    help="file 模式按这个频率节流（0=不节流，一秒钟灌完）")
    ap.add_argument("--payload", choices=["flat", "noise"], default="flat",
                    help="合成图类型：flat 纯色块（负载小）/ noise 噪声图（压不动，用来复现下游卡死）")
    ap.add_argument("--call-timeout", type=float, default=10.0, dest="call_timeout")
    ap.add_argument("--total-timeout", type=float, default=0.0, dest="total_timeout")
    args = ap.parse_args()

    global PAYLOAD
    PAYLOAD = args.payload

    if args.mode == "unit":
        return mode_unit(args)
    if args.mode == "blackhole":
        return mode_blackhole(args)
    if args.mode == "timing":
        return mode_timing(args)
    if args.mode == "file":
        return mode_file(args)
    return mode_compare(args)


if __name__ == "__main__":
    sys.exit(main())
