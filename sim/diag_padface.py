"""Where on the finger does the cube actually get held?

If the pad faces are not parallel to the sides of the cube, the jaw meets it on
a corner and the grasp wedges. The fingers' inner faces are extracted here as
the vertices nearest the other finger, in the gripper_base frame, and the
resulting plane is compared with the axis the fingers close along.

    python diag_padface.py
"""
import glob

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
GRASP = 270
np.set_printoptions(precision=4, suppress=True)

df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
base = m.body("right_gripper_base").id
cube = m.body("cube").id
j7 = int(np.ravel(m.joint("right_joint7").qposadr)[0])
j8 = int(np.ravel(m.joint("right_joint8").qposadr)[0])

for i in range(1, 7):
    d.qpos[int(np.ravel(m.joint(f"right_joint{i}").qposadr)[0])] = (
        SIGN[i] * float(actions[GRASP][6 + i]))
d.qpos[j7] = float(actions[GRASP][13]) / 2
d.qpos[j8] = -float(actions[GRASP][13]) / 2
mujoco.mj_forward(m, d)

R = d.xmat[base].reshape(3, 3)
o = d.xpos[base]
cube_local = (d.xpos[cube] - o) @ R
print("cube centre in gripper_base frame", np.round(cube_local * 1000, 1), "mm")
print("cube size 30.8 mm")


def mesh_in_base(body):
    bid = m.body(body).id
    pts = []
    for g in range(m.body_geomadr[bid], m.body_geomadr[bid] + m.body_geomnum[bid]):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        h = m.geom_dataid[g]
        a, n = m.mesh_vertadr[h], m.mesh_vertnum[h]
        Rb = d.xmat[bid].reshape(3, 3)
        pts.append((m.mesh_vert[a:a + n] @ Rb.T + d.xpos[bid] - o) @ R)
    return np.concatenate(pts)


for body, side in (("right_link7", -1), ("right_link8", +1)):
    p = mesh_in_base(body) * 1000
    # the inner face is the part of the mesh nearest the cube centre line
    inner = p[np.abs(p[:, 1] - cube_local[1] * 1000) < 10]
    if len(inner) < 10:
        inner = p
    q = p[np.abs(p[:, 2] - cube_local[2] * 1000) < 20]   # near the cube in z
    print(f"\n{body}: {len(p)} verts, {len(q)} at cube height")
    if len(q):
        print(f"  y range at cube height [{q[:,1].min():7.2f}, {q[:,1].max():7.2f}] mm")
        # innermost 15% of the vertices at cube height. link7 sits on the
        # minus-y side, so its inner face is the one with the larger y.
        edge = np.percentile(q[:, 1], 85 if side < 0 else 15)
        pad = q[q[:, 1] >= edge] if side < 0 else q[q[:, 1] <= edge]
        print(f"  inner face: {len(pad)} verts  y mean {pad[:,1].mean():7.2f} mm "
              f"sd {pad[:,1].std():.2f}")
        for z0 in np.arange(-40, 41, 20):
            band = pad[np.abs(pad[:, 2] - (cube_local[2] * 1000 + z0)) < 8]
            if len(band) > 3:
                print(f"    z {z0:+5.1f} mm rel cube: y mean {band[:,1].mean():8.2f} "
                      f"min {band[:,1].min():8.2f} max {band[:,1].max():8.2f}")
