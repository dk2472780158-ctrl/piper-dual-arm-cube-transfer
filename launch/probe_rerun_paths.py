#!/usr/bin/env python3
"""逼出 Rerun 里真实的实体路径 —— 不猜、不照抄，跑一遍抓出来。

做法：把 rerun.log 猴补丁掉，收集所有被写入的 entity path；
然后喂给 log_rerun_data 的是**按 piper_dual.get_observation() 原样构造**的
观测字典（裸相机名 left/middle/right + 28 个独立浮点键），action 走项目
真正的 make_robot_action + 项目自己的 features()。

不连机械臂、不开相机、不开 CAN。

用法（先 source rerun_env.sh）：
    python -B probe_rerun_paths.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "resident_new_env_rerun.py")
H, W = 480, 640


def load_features():
    """借用派生 runner 里的 features()，保证名字与真机一致。"""
    spec = importlib.util.spec_from_file_location("_rne", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.features()


def main() -> int:
    import torch
    import rerun as rr

    from lerobot.policies.utils import make_robot_action
    from lerobot.utils import visualization_utils as vu

    captured: list[str] = []

    def fake_log(entity_path=None, entity=None, *a, **k):
        captured.append(str(entity_path))

    rr.log = fake_log  # log_rerun_data 走的是这个

    feats = load_features()
    act_names = feats["action"]["names"]
    print(f"[paths] features()['action']['names'] 共 {len(act_names)} 个：")
    print("        " + ", ".join(act_names))
    print()

    # action：走项目真正的 make_robot_action
    action = make_robot_action(torch.zeros(1, len(act_names)), feats)
    print(f"[paths] make_robot_action -> dict，{len(action)} 项，键形如 {list(action)[:2]}")

    # observation：按 piper_dual.get_observation() 的原样构造
    #   - 关节：obs_dict[f"{side}_{joint}.{pos|effort}"] = float
    #   - 相机：obs_dict[name] = frame（裸名，uint8 HWC）
    obs: dict = {}
    for side in ("left", "right"):
        for joint in [*(f"joint_{i}" for i in range(1, 7)), "gripper"]:
            for kind in ("pos", "effort"):
                obs[f"{side}_{joint}.{kind}"] = 0.0
    for name in ("left", "middle", "right"):
        obs[name] = np.zeros((H, W, 3), dtype=np.uint8)

    print(f"[paths] 构造的 observation：{len(obs)} 个键"
          f"（28 个关节浮点 + 3 个相机数组）")
    print()

    vu.log_rerun_data(observation=obs, action=action, compress_images=True)

    uniq = sorted(set(captured))
    scalars = [p for p in uniq if not p.startswith("observation.left\x00")]
    print(f"[paths] log_rerun_data 一共写入 {len(captured)} 次，去重后 {len(uniq)} 个实体：")
    for p in uniq:
        print("        " + p)
    print()
    print(f"[paths] PATH_COUNT={len(uniq)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
