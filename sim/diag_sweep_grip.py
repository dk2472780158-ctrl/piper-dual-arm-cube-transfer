"""Which pad friction and cube mass carry the cube to the yellow mat?

The replay loses the cube by a slow downward slide through the pads, so the
numbers that decide it are the coefficient between pad and cube and the mass
being held. Both are swept together against the recorded demonstration.

    python diag_sweep_grip.py
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


def replay(mu, mass):
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    m.body_mass[m.body("cube").id] = mass
    m.geom_friction[m.geom("cube_geom").id][0] = mu
    for side in ("left", "right"):
        for body in ("link7", "link8"):
            bid = m.body(f"{side}_{body}").id
            for g in range(m.body_geomadr[bid],
                           m.body_geomadr[bid] + m.body_geomnum[bid]):
                m.geom_friction[g][0] = mu
    cube = m.body("cube").id
    for i in range(1, 7):
        d.qpos[int(np.ravel(m.joint(f"left_joint{i}").qposadr)[0])] = (
            SIGN[i] * LEFT_START[i - 1])
        d.ctrl[m.actuator(f"left_act_{i}").id] = SIGN[i] * LEFT_START[i - 1]
    for k in (7, 8):
        d.qpos[int(np.ravel(m.joint(f"left_joint{k}").qposadr)[0])] = 0.0
    mujoco.mj_forward(m, d)
    start = d.xpos[cube].copy()
    lift = 0.0
    for f in range(len(actions)):
        for i in range(1, 7):
            d.ctrl[m.actuator(f"right_act_{i}").id] = (
                SIGN[i] * float(actions[f][6 + i]))
        g = float(np.clip(actions[f][13], 0.0, 0.07))
        d.ctrl[m.actuator("right_act_7").id] = g / 2
        d.ctrl[m.actuator("right_act_8").id] = -g / 2
        for _ in range(17):
            mujoco.mj_step(m, d)
        lift = max(lift, float(d.xpos[cube][2] - start[2]))
    end = d.xpos[cube].copy()
    return (100 * lift, 100 * float(np.linalg.norm(end[:2] - start[:2])),
            bool(end[1] > 0.05), end)


print(f"{'mu':>5} {'mass g':>7} {'max lift':>9} {'travel':>7} {'on yellow':>9}  end")
for mu in (1.4, 2.0, 3.0):
    for mass in (0.030, 0.020, 0.012):
        lift, travel, yellow, end = replay(mu, mass)
        print(f"{mu:5.1f} {1000 * mass:7.0f} {lift:9.2f} {travel:7.1f} "
              f"{str(yellow):>9}  {np.round(end, 3)}")
