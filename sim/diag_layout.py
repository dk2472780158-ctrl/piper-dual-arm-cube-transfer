"""Replay the *recorded* actions in the sim and check where the arms land.

Reads episode 0 of the real r2l dataset, feeds its action column into the same
scene the policy runs in, and reports the world position of each gripper plus
the cube. Renders the middle camera at a few frames for side-by-side comparison
with the real dataset frames.
"""
import glob
import os

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
OUT = "/tmp/layout"

SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)

path = sorted(glob.glob(REAL))[0]
df = pd.read_parquet(path)
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)
states = np.stack(df["observation.state"].values)
print("episode frames", len(df))

os.makedirs(OUT, exist_ok=True)
renderer = mujoco.Renderer(m, 480, 640)


def place(pose14):
    for side, base in (("left", 0), ("right", 7)):
        for i in range(1, 7):
            j = m.joint(f"{side}_joint{i}")
            d.qpos[int(np.ravel(j.qposadr)[0])] = SIGN[i] * float(pose14[base + i - 1])
        grip = float(np.clip(pose14[base + 6], 0.0, 0.07))
        d.qpos[int(np.ravel(m.joint(f"{side}_joint7").qposadr)[0])] = grip / 2
        d.qpos[int(np.ravel(m.joint(f"{side}_joint8").qposadr)[0])] = -grip / 2
    mujoco.mj_forward(m, d)


def tip(side):
    """Middle point between the two fingertips, world coords."""
    pts = []
    for g in (f"{side}_link7", f"{side}_link8"):
        bid = m.body(g).id
        pts.append(d.xpos[bid].copy())
    return np.mean(pts, axis=0)


cube = m.body("cube").id
print("table top 0.72, cube spawn", np.round(d.xpos[cube], 4))
for f in (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, len(actions) - 1):
    if f >= len(actions):
        continue
    place(actions[f])
    r = tip("right")
    l = tip("left")
    c = d.xpos[cube]
    print(f"f{f:4d}  R tip {np.round(r, 3)}  L tip {np.round(l, 3)}  "
          f"cube {np.round(c, 3)}  |R-cube| {np.linalg.norm(r - c):.3f}")
    renderer.update_scene(d, camera="middle")
    import imageio.v2 as imageio
    imageio.imwrite(f"{OUT}/actionmiddle_{f:04d}.png", renderer.render())
print("wrote", OUT)
