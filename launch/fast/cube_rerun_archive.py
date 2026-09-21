#!/usr/bin/env python3
"""把 Rerun 录像同时落一份到**本地盘**，另外通过软链接对外可见。

为什么落本地盘、不直接写网络工作区
----------------------------------
网络挂载的工作区每写一个字节都要上行。一次 15 秒的回合是几 MB 到几十 MB 的
持续写入，会和 30 Hz 的控制链路抢带宽——表现为控制循环抖动。所以录像写到
`CUBE_RERUN_SAVE_DIR`（默认 `$PIPER_WS/rerun_data`，本地盘）。

怎么接的（关键：set_sinks 是"替换"语义）
----------------------------------------
rerun 0.31.3 的 rr.set_sinks() 文档原话：先 connect_grpc 再 set_sinks，**只会留下
set_sinks 里那几个 sink**。所以不能在 init_rerun 之后单独调 rr.save() —— 那会把
实时那条 gRPC sink 顶掉，查看器就没数据了。正确做法是一次性给两个：

    rr.set_sinks(rr.FileSink(本地文件), rr.GrpcSink(f"rerun+http://{ip}:{port}/proxy"))

两条同时活：查看器照旧实时显示，磁盘上多一份完整录像。

必须在 piper_cube_record 被 import **之前**装好 —— 那个文件用的是
``from ... import init_rerun``（按名字绑定，piper_cube_record.py:156）。由
sitecustomize.py 在解释器启动时装，时机满足。

开关
----
    CUBE_RERUN_SAVE=off       完全不装（一键回退，行为与改动前一致）
    CUBE_RERUN_SAVE_DIR=...   存到哪；没设就整个不落盘
    CUBE_RERUN_SAVE_META=off  不写同名的 .cmd.txt

边界：写失败/目录不可写时只打印一行并**保持原行为**（宁可没有录像，也不能让面板
没数据，更不能影响控制循环）。本模块不 import rerun，直到 init_rerun 真被调用。
"""
from __future__ import annotations

import atexit
import os
import re
import sys
import time

__all__ = ["ArchiveRecorder", "install"]

_DISABLE_VALUES = ("0", "off", "false", "no")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_SECRET = re.compile(r"(?i)(token|password|passwd|secret|api[_-]?key|credential)")

DEFAULT_SESSION = "lerobot_control_loop"


def _is_off(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in _DISABLE_VALUES


def _sanitize(text, limit: int = 80) -> str:
    cleaned = _UNSAFE.sub("_", str(text)).strip("_")
    return cleaned[:limit] or "run"


def _run_name(session_name) -> str:
    """优先用 --dataset.repo_id（去掉 local/），那是这一次跑的真正身份。

    init_rerun 的 session_name 是固定的 "recording"，光靠它分不清是哪次跑。
    """
    for arg in sys.argv[1:]:
        if arg.startswith("--dataset.repo_id="):
            value = arg.split("=", 1)[1]
            if value.startswith("local/"):
                value = value[len("local/"):]
            return _sanitize(value)
    return _sanitize(session_name or DEFAULT_SESSION)


def _scrub(argv):
    """命令行里如果有 --token/--password 之类，只留参数名。"""
    out = []
    for arg in argv:
        key = arg.split("=", 1)[0] if "=" in arg else ""
        out.append(f"{key}=<已隐去>" if key and _SECRET.search(key) else arg)
    return out


class ArchiveRecorder:
    """可调用对象，签名与 init_rerun 完全一致，所以能直接替换掉它。"""

    def __init__(self, original_init_rerun, save_dir: str):
        self._original = original_init_rerun
        self.save_dir = save_dir
        self.path: str | None = None
        self._reported = False

    def __call__(self, *args, **kwargs):
        # 先照原样跑：rr.init + 连平台那条 gRPC。这一步出问题不该由本模块负责。
        self._original(*args, **kwargs)
        try:
            self._attach(args, kwargs)
        except Exception as exc:  # noqa: BLE001  录像失败绝不能影响业务
            print(
                f"[cube_rerun_save] 录像未开启（面板照旧）：{type(exc).__name__}: {exc}",
                flush=True,
            )

    def _attach(self, args, kwargs):
        session_name = kwargs.get("session_name", args[0] if args else DEFAULT_SESSION)
        ip = kwargs.get("ip", args[1] if len(args) > 1 else None)
        port = kwargs.get("port", args[2] if len(args) > 2 else None)
        if not (ip and port):
            # 原函数在没给 ip/port 时走 rr.spawn()，sink 由 spawn 自己管，不掺和。
            print("[cube_rerun_save] 没给 ip/port（spawn 分支）—— 不落录像", flush=True)
            return

        import rerun as rr

        os.makedirs(self.save_dir, exist_ok=True)
        self.path = self._pick_path(_run_name(session_name))
        # 先构造 FileSink：目录不可写会在这里抛，此时还没动过 sink，面板不受影响。
        file_sink = rr.FileSink(self.path)
        try:
            rr.set_sinks(file_sink, rr.GrpcSink(f"rerun+http://{ip}:{port}/proxy"))
        except Exception:
            # 兜底：把平台那条单独接回去，宁可丢录像也不让面板空掉。
            rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
            raise
        print(f"[cube_rerun_save] 录像 -> {self.path}", flush=True)
        if not _is_off("CUBE_RERUN_SAVE_META"):
            self._write_meta()

    def _pick_path(self, run: str) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = os.path.join(self.save_dir, f"{run}_{stamp}")
        path = f"{base}.rrd"
        suffix = 1
        while os.path.exists(path):  # 同一秒起了两个（循环脚本重入）就加序号
            path = f"{base}-{suffix}.rrd"
            suffix += 1
        return path

    def _write_meta(self) -> None:
        try:
            with open(self.path[:-len(".rrd")] + ".cmd.txt", "w", encoding="utf-8") as fh:
                fh.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                fh.write(f"目录: {os.getcwd()}\n")
                fh.write("命令: " + " ".join(_scrub(sys.argv)) + "\n")
        except Exception:  # noqa: BLE001  附带件而已
            pass

    def report(self) -> None:
        """退出时收尾：先把 SDK 里没落盘的刷下去，再报文件大小。"""
        if self._reported or not self.path:
            return
        self._reported = True
        try:
            import rerun_bindings

            rerun_bindings.flush(timeout_sec=5)
        except Exception:  # noqa: BLE001  拿不到就算了，文件里已有前面刷下去的部分
            pass
        try:
            size = os.path.getsize(self.path)
        except OSError:
            size = 0
        print(f"[cube_rerun_save] 录像收尾 {self.path}（{size / 1e6:.1f} MB）", flush=True)


def install(original_init_rerun):
    """装录像钩子。关掉或没配目录时，原样返回，行为与改动前完全一致。"""
    if _is_off("CUBE_RERUN_SAVE"):
        print("[cube_rerun_save] CUBE_RERUN_SAVE=off —— 不落录像，面板照旧", flush=True)
        return original_init_rerun
    save_dir = os.environ.get("CUBE_RERUN_SAVE_DIR", "").strip()
    if not save_dir:
        print("[cube_rerun_save] 未设 CUBE_RERUN_SAVE_DIR —— 不落录像", flush=True)
        return original_init_rerun

    recorder = ArchiveRecorder(original_init_rerun, save_dir)
    # 注册得**早于** cube_fastlog 的收尾：atexit 是后进先出，
    # 这样顺序是"快版先把队列发完 → 再刷盘 + 报大小"。
    atexit.register(recorder.report)
    return recorder
