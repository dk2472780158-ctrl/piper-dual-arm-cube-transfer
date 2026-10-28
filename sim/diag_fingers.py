"""Where the finger meshes actually sit, in the arm base frame.

The TCP is not a body origin, so the grasp height is decided from the finger
mesh geometry instead: the pads have to bracket the cube vertically and meet
it in x/y. Printed for the recorded grasp and release frames.
"""
import glob

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
np.set_printoptions(precision=4, suppress=True)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)

mount_id = m.body("right_mount").id


def place(row14):
    for i in range(1, 7):
        d.qpos[int(np.ravel(m.joint(f"right_joint{i}").qposadr)[0])] = (
            SIGN[i] * float(row14[6 + i]))
    g = float(np.clip(row14[13], 0.0, 0.07))
    d.qpos[int(np.ravel(m.joint("right_joint7").qposadr)[0])] = g / 2
    d.qpos[int(np.ravel(m.joint("right_joint8").qposadr)[0])] = -g / 2
    mujoco.mj_forward(m, d)


def mesh_points(body):
    bid = m.body(body).id
    pts = []
    for g in range(m.body_geomadr[bid], m.body_geomadr[bid] + m.body_geomnum[bid]):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        h = m.geom_dataid[g]
        a, n = m.mesh_vertadr[h], m.mesh_vertnum[h]
        v = m.mesh_vert[a:a + n]
        R = d.xmat[bid].reshape(3, 3)
        pts.append(v @ R.T + d.xpos[bid])
    return np.concatenate(pts)


for label, f in (("grasp", 185), ("release", 376)):
    place(actions[f])
    Rm = d.xmat[mount_id].reshape(3, 3)
    o = d.xpos[mount_id]
    both = []
    for body in ("right_link7", "right_link8"):
        p = mesh_points(body)
        local = (p - o) @ Rm
        both.append(local)
        print(f"{label} {body}: x [{local[:,0].min():.3f},{local[:,0].max():.3f}] "
              f"y [{local[:,1].min():.3f},{local[:,1].max():.3f}] "
              f"z [{local[:,2].min():.3f},{local[:,2].max():.3f}]")
    local = np.concatenate(both)
    gap_axis = 1  # fingers separate along the base y axis according to the bounds
    print(f"{label} pads  x [{local[:,0].min():.3f},{local[:,0].max():.3f}] "
          f"y [{local[:,1].min():.3f},{local[:,1].max():.3f}] "
          f"z [{local[:,2].min():.3f},{local[:,2].max():.3f}]")
    # darkest inner faces: the pad centre is the midpoint of the open span
    print(f"{label} pad centre x {(local[:,0].min()+local[:,0].max())/2:.4f} "
          f"y {(local[:,1].min()+local[:,1].max())/2:.4f} "
          f"z {(local[:,2].min()+local[:,2].max())/2:.4f}")
    print()
