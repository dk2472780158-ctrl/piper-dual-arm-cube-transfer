#!/usr/bin/env python3
"""解释器启动时由 site 自动 import（前提：本目录在 PYTHONPATH 里，见 fast_env.sh）。

两件事，都在 ``lerobot.scripts.piper_cube_record`` 被 import **之前**做完
—— 那个文件用的是 ``from ... import init_rerun, log_rerun_data``（按名字绑定，
见 piper_cube_record.py:156），晚于它再替换就白换了。

1. 把 ``lerobot.utils.visualization_utils.log_rerun_data`` 换成快版（cube_fastlog.py）。
2. 把 ``...init_rerun`` 包一层，顺带把录像落一份到本地盘（cube_rerun_archive.py）。

第 2 件**先注册**：atexit 后进先出，这样退出顺序是"快版把队列发完 → 再刷盘"。

安全底线：任何异常都只打印一行，然后**按原行为继续**。监控不能搞崩业务。
"""
from __future__ import annotations

import os
import sys

_DISABLE_VALUES = ("0", "off", "false", "no")


def _install() -> None:
    if os.environ.get("CUBE_RERUN_FAST", "1").strip().lower() in _DISABLE_VALUES:
        print("[cube_rerun_fast] CUBE_RERUN_FAST=off —— 不装补丁，行为与原来完全一致", flush=True)
        return

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    import lerobot.utils.visualization_utils as viz

    # 双保险用的：万一本模块是在 piper_cube_record 之后才被 import 的
    core = sys.modules.get("lerobot.scripts.piper_cube_record")

    # --- 1) 录像落本地盘（先装，让它的收尾在快版之后跑）---
    import cube_rerun_archive

    init_rerun = cube_rerun_archive.install(viz.init_rerun)
    viz.init_rerun = init_rerun
    if core is not None:
        core.init_rerun = init_rerun

    # --- 2) log_rerun_data 换成快版 ---
    import cube_fastlog

    original = viz.log_rerun_data
    logger = cube_fastlog.install(original)
    viz.log_rerun_data = logger
    # 留给验证脚本取原件用（probe_fastlog.py 需要拿原函数做对照）
    viz._cube_rerun_fast_original = original
    if core is not None:
        core.log_rerun_data = logger

    print(f"[cube_rerun_fast] 已启用 {logger.describe()}", flush=True)
    if isinstance(init_rerun, cube_rerun_archive.ArchiveRecorder):
        print(f"[cube_rerun_save] 录像目录 {init_rerun.save_dir}", flush=True)


try:
    _install()
except Exception as exc:  # noqa: BLE001  补丁失败绝不能影响业务
    print(f"[cube_rerun_fast] 补丁未生效（按原行为继续）：{type(exc).__name__}: {exc}", flush=True)
