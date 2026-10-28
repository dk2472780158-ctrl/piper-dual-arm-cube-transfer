"""Where does the simulated right gripper go as the recorded arm reaches?

The wrist cameras sit on the gripper, so any error in the arm's geometry shows
up as the cube being the wrong size in those views. The middle view cannot say
which is wrong -- a finger that is too low and a link that is too long look the
same. This prints the fingertip and its distance to the cube, which separates
the two: the fingertips are supposed to arrive at the cube, and at the grasp
frame they are supposed to be around it.

Usage:  MUJOCO_GL=egl python diag_tip_track.py
"""
import glob

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
DATASET = ("/home/databall/.cache/huggingface/lerobot/local/"
           "cube_r2l_dataset_v2_robust")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
np.set_printoptions(precision=4, suppress=True)

df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(
    f"{DATASET}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)


def tip(side):
    """Middle point between the two fingertips, world coords."""
    return np.mean([d.xpos[m.body(f"{side}_link{g}").id].copy() for g in (7, 8)], axis=0)


cube = m.body("cube").id
print("frame  right tip            cube              gap m   tip z - table")
for f in range(0, len(actions), 15):
    pose = actions[f]
    for side, base in (("left", 0), ("right", 7)):
        for i in range(1, 7):
            d.qpos[int(np.ravel(m.joint(f"{side}_joint{i}").qposadr)[0])] = (
                SIGN[i] * float(pose[base + i - 1]))
        grip = float(np.clip(pose[base + 6], 0.0, 0.07))
        d.qpos[int(np.ravel(m.joint(f"{side}_joint7").qposadr)[0])] = grip / 2
        d.qpos[int(np.ravel(m.joint(f"{side}_joint8").qposadr)[0])] = -grip / 2
    mujoco.mj_forward(m, d)
    point, block = tip("right"), d.xpos[cube].copy()
    print(f"{f:5d}  {np.round(point, 3)}  {np.round(block, 3)}  "
          f"{np.linalg.norm(point - block):.3f}   {point[2] - 0.712:+.3f}")
