"""Is the gripper closing on the cube's faces or across its corners?

A parallel jaw that closes 20 deg off the cube's face normal wedges itself on
two corners. That grasp survives a slow lift and lets go as soon as the arm
swings. The cube's yaw in this scene is fitted from the recorded middle frame;
the gripper's yaw comes from the recorded joints and the arm base yaw. The
angle between the two is what this prints, at the grasp and the release frame.

    python diag_yaw.py
"""
import glob
import math

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
np.set_printoptions(precision=4, suppress=True)

df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
b7 = m.body("right_link7").id
b8 = m.body("right_link8").id
cube = m.body("cube").id
j7 = int(np.ravel(m.joint("right_joint7").qposadr)[0])
j8 = int(np.ravel(m.joint("right_joint8").qposadr)[0])


def heading(v):
    return math.degrees(math.atan2(v[1], v[0]))


for f in (240, 255, 270, 376):
    for i in range(1, 7):
        d.qpos[int(np.ravel(m.joint(f"right_joint{i}").qposadr)[0])] = (
            SIGN[i] * float(actions[f][6 + i]))
    d.qpos[j7] = 0.035
    d.qpos[j8] = -0.035
    mujoco.mj_forward(m, d)
    # closing axis = the direction the two fingers separate along
    closing = d.xpos[b7] - d.xpos[b8]
    closing[2] = 0.0
    # cube face normal: its local x axis in world
    R = d.xmat[cube].reshape(3, 3)
    normal = R[:, 0].copy()
    normal[2] = 0.0
    diff = (heading(closing) - heading(normal) + 90) % 180 - 90
    print(f"f{f:4d}  closing {heading(closing):+7.2f} deg  "
          f"cube face normal {heading(normal):+7.2f} deg  "
          f"offset {diff:+7.2f} deg")
