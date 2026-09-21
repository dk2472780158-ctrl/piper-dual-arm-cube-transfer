#!/usr/bin/env python3
"""无硬件 Rerun 链路探针 —— 证明 SDK 能把数据送进 9876 服务端。

它复用项目自己的 visualize 函数（init_rerun / log_rerun_data），但喂的是
**合成数据**：键名和结构**照真机抄**（见 probe_rerun_paths.py 抓出来的清单），
所以页面上出现的实体树形状与真机一致，只有内容是假的（纯色块 + 正弦）。

不连机械臂、不开相机、不开 CAN、不读 /dev/video*。

用法（必须先 source rerun_env.sh）：
    source launch/rerun_env.sh
    python -B launch/probe_rerun.py

合成画面是纯色（左=红 中=绿 右=蓝），一眼能认出不是真实相机。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

SESSION = os.environ.get("CUBE_PROBE_SESSION", "cube_probe")
N_STEPS = 200
DT = 0.05
H, W = 480, 640

SIDES = ("left", "right")
JOINTS = [*(f"joint_{i}" for i in range(1, 7)), "gripper"]


def synthetic_observation(step: int) -> dict:
    """按 piper_dual.get_observation() 的结构造一份观测。"""
    t = step / 30.0
    obs: dict = {}
    for side in SIDES:
        for j, joint in enumerate(JOINTS):
            phase = t + j * 0.2 + (0.9 if side == "right" else 0.0)
            obs[f"{side}_{joint}.pos"] = float(np.sin(phase))
            obs[f"{side}_{joint}.effort"] = float(0.3 * np.cos(phase))
    # 三路纯色合成帧：左=红 中=绿 右=蓝
    for channel, name in enumerate(("left", "middle", "right")):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[..., channel] = 200
        obs[name] = img
    return obs


def synthetic_action(step: int) -> dict:
    """按 make_robot_action 的键名造一份动作。"""
    t = step / 30.0
    action: dict = {}
    for side in SIDES:
        for j, joint in enumerate(JOINTS):
            action[f"{side}_{joint}.pos"] = float(np.sin(t + j * 0.2))
    return action


def main() -> int:
    import rerun

    print(f"[probe] rerun SDK {rerun.__version__}", flush=True)
    print(f"[probe]   {rerun.__file__}", flush=True)
    if not rerun.__version__.startswith("0.31"):
        print(
            "[probe] 警告：版本不是 0.31.x —— PYTHONPATH 可能没生效，"
            "这个版本和 visualization_utils.py 不兼容",
            flush=True,
        )

    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    ip = os.environ.get("CUBE_RERUN_IP", "127.0.0.1")
    port = int(os.environ.get("CUBE_RERUN_PORT", "9876"))
    print(f"[probe] init_rerun session={SESSION} -> rerun+http://{ip}:{port}/proxy", flush=True)
    init_rerun(session_name=SESSION, ip=ip, port=port)

    for step in range(N_STEPS):
        log_rerun_data(
            observation=synthetic_observation(step),
            action=synthetic_action(step),
            compress_images=True,
        )
        if step % 40 == 0:
            print(f"[probe] step={step}", flush=True)
        time.sleep(DT)

    time.sleep(2.0)  # 等 SDK flush 完再退出
    print(f"[probe] PROBE_DONE steps={N_STEPS}（画面是纯色合成帧，不是真实相机）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
