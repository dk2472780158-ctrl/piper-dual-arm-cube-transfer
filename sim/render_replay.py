"""Render the replay at a few frames so the views can be compared to the data."""
import glob
import os

import numpy as np
import pandas as pd
import mujoco
import imageio.v2 as imageio

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
OUT = "/tmp/replay"
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
LEFT_START = np.asarray([0.0093, 0.0049, -0.0024, -0.0582, 0.3185, 0.0973, 0.0003])
SHOTS = (0, 150, 270, 300, 330, 390)

os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, 480, 640)


def drive(pose7):
    for i in range(1, 7):
        d.ctrl[m.actuator(f"right_act_{i}").id] = SIGN[i] * float(pose7[i - 1])
    g = float(np.clip(pose7[6], 0.0, 0.07))
    d.ctrl[m.actuator("right_act_7").id] = g / 2
    d.ctrl[m.actuator("right_act_8").id] = -g / 2


mujoco.mj_resetData(m, d)
for i in range(1, 7):
    d.qpos[int(np.ravel(m.joint(f"left_joint{i}").qposadr)[0])] = (
        SIGN[i] * LEFT_START[i - 1])
    d.ctrl[m.actuator(f"left_act_{i}").id] = SIGN[i] * LEFT_START[i - 1]
for k in (7, 8):
    d.qpos[int(np.ravel(m.joint(f"left_joint{k}").qposadr)[0])] = 0.0
mujoco.mj_forward(m, d)

for f in range(len(actions)):
    drive(actions[f][7:14])
    for _ in range(17):
        mujoco.mj_step(m, d)
    if f in SHOTS:
        for cam in ("middle", "left", "right"):
            renderer.update_scene(d, camera=cam)
            imageio.imwrite(f"{OUT}/f{f:04d}_{cam}.png", renderer.render())
        print("frame", f, "cube", np.round(d.xpos[m.body("cube").id], 3))
print("wrote", OUT)
