"""Solve the finger aperture offset against the recorded touch point.

The recorded hold has the fingers touching the cube at a joint reading of
0.0334. Here the touch point is measured by bisection for a range of finger
offsets, so the offset that reproduces the hardware can be read off instead of
guessed.

    python diag_delta.py
"""
import glob

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
TARGET = 0.0334
GRASP = 255
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
np.set_printoptions(precision=4, suppress=True)

df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
POSE = np.asarray(pd.Series(df["action"].values).iloc[GRASP])[7:13]

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
b7 = m.body("right_link7").id
b8 = m.body("right_link8").id
cube = m.body("cube").id
j7 = int(np.ravel(m.joint("right_joint7").qposadr)[0])
j8 = int(np.ravel(m.joint("right_joint8").qposadr)[0])


def place(offset):
    """Put the arm at the grasp pose with the fingers `offset` apart wider."""
    m.body_pos[b7][1] = -offset
    m.body_pos[b8][1] = offset


def touched(q):
    d.qpos[j7] = q / 2
    d.qpos[j8] = -q / 2
    mujoco.mj_forward(m, d)
    for k in range(d.ncon):
        c = d.contact[k]
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                               m.geom_bodyid[c.geom1]) or "world"
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                               m.geom_bodyid[c.geom2]) or "world"
        if "cube" in (n1, n2) and (n1, n2) != ("cube", "cube"):
            return True
    return False


for i in range(1, 7):
    d.qpos[int(np.ravel(m.joint(f"right_joint{i}").qposadr)[0])] = SIGN[i] * POSE[i - 1]

print(f"{'offset':>9} {'touch q':>9} {'vs 0.0334':>10}")
for offset in np.arange(-0.006, 0.0061, 0.002):
    place(offset)
    lo, hi = 0.001, 0.069        # lo touches, hi does not
    if not touched(lo) or touched(hi):
        print(f"{offset:+9.4f}   ({'no contact' if not touched(lo - 0.02) else 'contact even when wide'})")
        continue
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if touched(mid):
            lo = mid
        else:
            hi = mid
    print(f"{offset:+9.4f} {lo:9.4f} {lo - TARGET:+10.4f}")
