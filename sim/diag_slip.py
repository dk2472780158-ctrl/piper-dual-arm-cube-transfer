"""The frame where the cube is lost, with the accelerations that caused it.

A grasp that holds at rest and lets go during the swing is a force budget
question: the pads can hold friction times the normal force, and the swing
asks for mass times acceleration. Both sides are printed here per frame.

    python diag_slip.py
"""
import glob

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
LEFT_START = np.asarray([0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222,
                         0.2780382633, 0.0])
F0, F1 = 270, 306
np.set_printoptions(precision=4, suppress=True)

df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
cube = m.body("cube").id
b7 = m.body("right_link7").id
b8 = m.body("right_link8").id

for i in range(1, 7):
    d.qpos[int(np.ravel(m.joint(f"left_joint{i}").qposadr)[0])] = (
        SIGN[i] * LEFT_START[i - 1])
    d.ctrl[m.actuator(f"left_act_{i}").id] = SIGN[i] * LEFT_START[i - 1]
for k in (7, 8):
    d.qpos[int(np.ravel(m.joint(f"left_joint{k}").qposadr)[0])] = 0.0
mujoco.mj_forward(m, d)

prev_p = prev_v = None
print(f"{'f':>4} {'finger mid':>24} {'speed m/s':>10} {'acc m/s^2':>10} "
      f"{'cube':>22} {'n':>2} {'F normal':>9}")
for f in range(len(actions)):
    for i in range(1, 7):
        d.ctrl[m.actuator(f"right_act_{i}").id] = SIGN[i] * float(actions[f][6 + i])
    g = float(np.clip(actions[f][13], 0.0, 0.07))
    d.ctrl[m.actuator("right_act_7").id] = g / 2
    d.ctrl[m.actuator("right_act_8").id] = -g / 2
    for _ in range(17):
        mujoco.mj_step(m, d)
    if f < F0 or f > F1:
        continue
    mid = 0.5 * (d.xpos[b7] + d.xpos[b8])
    p = mid.copy()
    if prev_p is not None:
        v = (p - prev_p) * 30.0
        a = (v - prev_v) * 30.0 if prev_v is not None else np.zeros(3)
        prev_v = v
    else:
        prev_p = p
        continue
    prev_p = p
    n, force = 0, 0.0
    for k in range(d.ncon):
        c = d.contact[k]
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                               m.geom_bodyid[c.geom1]) or "world"
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                               m.geom_bodyid[c.geom2]) or "world"
        if "cube" in (n1, n2) and "link" in (n1 + n2):
            buf = np.zeros(6)
            mujoco.mj_contactForce(m, d, k, buf)
            n += 1
            force += abs(buf[0])
    print(f"{f:4d} {str(np.round(mid, 3)):>24} {np.linalg.norm(v):10.3f} "
          f"{np.linalg.norm(a):10.2f} {str(np.round(d.xpos[cube], 3)):>22} "
          f"{n:2d} {force:9.2f}")
