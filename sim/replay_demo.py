"""Replay the recorded demonstration in the sim and see whether the cube moves.

This is the replica test: if the sim cell is faithful, feeding the recorded
action column frame by frame should carry the cube from the purple mat to the
yellow mat, with no policy involved.
"""
import glob
import sys

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
LEFT_START = np.asarray([0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222,
                         0.2780382633, 0.0])
np.set_printoptions(precision=4, suppress=True)

df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
cube = m.body("cube").id


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

start = d.xpos[cube].copy()
print("spawn", np.round(start, 4))
for f in range(len(actions)):
    drive(actions[f][7:14])
    for _ in range(17):          # 1/30 s per recorded frame
        mujoco.mj_step(m, d)
    if f % 30 == 0:
        print(f"f{f:4d} cube {np.round(d.xpos[cube], 3)}")

end = d.xpos[cube].copy()
print("end  ", np.round(end, 4), "travel %.1f cm" % (100 * np.linalg.norm(end - start)))
print("lift %.1f cm" % (100 * (end[2] - start[2])))
