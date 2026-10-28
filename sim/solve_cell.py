"""Solve the cell layout from the recorded pick and place poses.

The SDK DH chain and the URDF chain are the same arm once joint 2 is offset by
-7.78 deg (see diag_fkfit.py), so MuJoCo can place the fingertips for any
recorded joint vector. The cell is then defined by one requirement: at the
grasp frame the fingertip sits on the purple mat, at the release frame it sits
on the yellow mat, and both mats lie on the table top.

Unknowns: right-arm base yaw and xyz, and the distance between the two mats.
The mats are assumed centred on x = 0 and symmetric about y = 0.
"""
import glob

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")

SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
DELTA = {}   # the URDF and DH zeroes agree once rpy is read as extrinsic
TABLE = 0.72
CUBE_HALF = 0.0225
MAT_T = 0.003
HZ = TABLE + MAT_T + CUBE_HALF
GRASP = 270
RELEASE = 385

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)


def tcp(row14):
    """Fingertip midpoint in the right mount frame."""
    for i in range(1, 7):
        j = m.joint(f"right_joint{i}")
        d.qpos[int(np.ravel(j.qposadr)[0])] = SIGN[i] * float(row14[6 + i]) + DELTA.get(i, 0.0)
    g = float(np.clip(row14[13], 0.0, 0.07))
    d.qpos[int(np.ravel(m.joint("right_joint7").qposadr)[0])] = g / 2
    d.qpos[int(np.ravel(m.joint("right_joint8").qposadr)[0])] = -g / 2
    mujoco.mj_forward(m, d)
    mid = 0.5 * (d.xpos[m.body("right_link7").id] + d.xpos[m.body("right_link8").id])
    Rm = d.xmat[m.body("right_mount").id].reshape(3, 3)
    return Rm.T @ (mid - d.xpos[m.body("right_mount").id])


pg = tcp(actions[GRASP])
pl = tcp(actions[RELEASE])
print("mount-frame fingertip at grasp  ", np.round(pg, 4))
print("mount-frame fingertip at release", np.round(pl, 4))
print("delta                             ", np.round(pl - pg, 4))

# For a candidate yaw and mat spacing, the best base offset is closed form.
best = None
for deg in np.arange(-180.0, 180.0, 0.25):
    yaw = np.radians(deg)
    Ry = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    rg, rl = Ry @ pg, Ry @ pl
    for sep in np.arange(0.14, 0.34, 0.002):
        tgt = np.array([[0.0, -sep / 2, HZ], [0.0, sep / 2, HZ]])
        src = np.array([rg, rl])
        # base offset: least squares over both poses
        b = (tgt - src).mean(axis=0)
        res = np.linalg.norm(src + b - tgt, axis=1)
        err = float((res ** 2).sum())
        if best is None or err < best[0]:
            best = (err, deg, sep, b.copy(), res)

err, deg, sep, b, res = best
print()
print(f"best: yaw {deg:+.2f} deg  mat separation {sep:.4f} m")
print("right base xyz", np.round(b, 4))
print("per-pose residual", np.round(res, 4))

yaw = np.radians(deg)
print()
print(f"CELL right_mount = ({b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}, {yaw:.6f})")
print(f"CELL left_mount  = ({b[0]:.4f}, {-b[1]:.4f}, {b[2]:.4f}, {-yaw:.6f})")
print(f"yellow_mat (0, {sep/2:.4f})  purple_mat (0, {-sep/2:.4f})")
