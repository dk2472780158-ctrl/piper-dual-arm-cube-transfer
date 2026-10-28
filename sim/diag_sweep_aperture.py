"""Sweep the finger aperture against the recorded demonstration.

Each candidate moves the two fingers apart by the given amount, replays the
recording, and reports how far the cube got. The candidates bracket the value
that reproduces the recorded 0.0334 stall; a wider aperture means the recorded
command presses harder.

    python diag_sweep_aperture.py
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
np.set_printoptions(precision=4, suppress=True)

df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)


def replay(aperture):
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    m.body_pos[m.body("right_link7").id][1] = aperture
    m.body_pos[m.body("right_link8").id][1] = -aperture
    cube = m.body("cube").id
    j7 = int(np.ravel(m.joint("right_joint7").qposadr)[0])
    j8 = int(np.ravel(m.joint("right_joint8").qposadr)[0])
    for i in range(1, 7):
        d.qpos[int(np.ravel(m.joint(f"left_joint{i}").qposadr)[0])] = (
            SIGN[i] * LEFT_START[i - 1])
        d.ctrl[m.actuator(f"left_act_{i}").id] = SIGN[i] * LEFT_START[i - 1]
    for k in (7, 8):
        d.qpos[int(np.ravel(m.joint(f"left_joint{k}").qposadr)[0])] = 0.0
    mujoco.mj_forward(m, d)
    start = d.xpos[cube].copy()
    hold, lift = [], 0.0
    for f in range(len(actions)):
        for i in range(1, 7):
            d.ctrl[m.actuator(f"right_act_{i}").id] = (
                SIGN[i] * float(actions[f][6 + i]))
        g = float(np.clip(actions[f][13], 0.0, 0.07))
        d.ctrl[m.actuator("right_act_7").id] = g / 2
        d.ctrl[m.actuator("right_act_8").id] = -g / 2
        for _ in range(17):
            mujoco.mj_step(m, d)
        if 270 <= f <= 300:
            hold.append(float(d.qpos[j7] - d.qpos[j8]))
        lift = max(lift, float(d.xpos[cube][2] - start[2]))
    end = d.xpos[cube].copy()
    return (float(np.mean(hold)), 100 * lift, 100 * float(np.linalg.norm(end[:2] - start[:2])),
            bool(end[1] > 0.05), end)


print(f"{'aperture':>9} {'open@hold':>10} {'max lift':>9} {'travel':>7} "
      f"{'on yellow':>9}  end")
for aperture in (0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004):
    open_hold, lift, travel, yellow, end = replay(aperture)
    print(f"{aperture:9.4f} {open_hold:10.4f} {lift:9.2f} {travel:7.1f} "
          f"{str(yellow):>9}  {np.round(end, 3)}")
