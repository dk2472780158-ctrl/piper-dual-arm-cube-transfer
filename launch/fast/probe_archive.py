#!/usr/bin/env python3
"""验证"录像落本地盘"这条链：**真实的 init_rerun**（被 sitecustomize 包过）+ 真服务端。

跟 probe_fastlog.py 的区别：那个用的是自己拼的 rr.FileSink，这个走的是生产那条路
（sitecustomize 在解释器启动时装好的 init_rerun 包装），所以能验出"两个 sink 是不是
真的同时活着"。

用法（必须在同一个 shell 里先 source launch/fast/fast_env.sh，否则补丁没装）：

  python -B probe_archive.py --port 9878 --steps 300

端口请指向**你自己起的**服务端（见 README 第七节的 C 验证），别连平台那个 9876。

退出码：0 正常；2 = 补丁没生效（环境没 source 对）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import probe_fastlog as pf  # noqa: E402  复用合成数据与统计


def main() -> int:
    ap = argparse.ArgumentParser(description="验证录像落盘链路")
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9878)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--payload", choices=["flat", "noise"], default="flat")
    args = ap.parse_args()

    pf.PAYLOAD = args.payload

    import cube_rerun_archive
    import lerobot.utils.visualization_utils as viz

    init_rerun = viz.init_rerun
    if not isinstance(init_rerun, cube_rerun_archive.ArchiveRecorder):
        print(
            "[probe] init_rerun 不是录像版 —— sitecustomize 没生效。"
            "先在同一个 shell 里 source launch/fast/fast_env.sh 再跑。",
            flush=True,
        )
        return 2

    print(f"[probe] 录像目录={init_rerun.save_dir} 目标={args.ip}:{args.port} "
          f"负载={args.payload} 步数={args.steps}", flush=True)
    init_rerun(session_name="probe_archive", ip=args.ip, port=args.port)

    logger = viz.log_rerun_data  # 生产链路上就是快版 logger
    t_start = time.perf_counter()
    for i in range(args.steps):
        logger(observation=pf.make_observation(i), action=pf.make_action(i))
        target = t_start + (i + 1) / args.hz
        now = time.perf_counter()
        if target > now:
            time.sleep(target - now)

    wall = time.perf_counter() - t_start
    print(f"[probe] {args.steps} 步，墙钟 {wall:.1f} s（{args.steps / wall:.1f} Hz）", flush=True)
    # 不调 os._exit：要让 atexit 里的收尾跑起来（刷盘 + 报文件大小）
    return 0


if __name__ == "__main__":
    sys.exit(main())
