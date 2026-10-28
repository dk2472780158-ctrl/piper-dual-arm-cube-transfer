"""The flat face of each finger, measured with the jaw closed.

With the joints at zero the two fingers meet, so the plane they meet on is the
gripping plane. The vertices that lie on it give its extent, and expressing
them in each finger's own frame says where a flat pad geom belongs. The mesh
alone has a chamfer there, which wedges the cube out of the jaw.

    python diag_padbox.py
"""
import numpy as np
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
np.set_printoptions(precision=4, suppress=True)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
base = m.body("right_gripper_base").id
for side in ("left", "right"):
    for k in (7, 8):
        d.qpos[int(np.ravel(m.joint(f"{side}_joint{k}").qposadr)[0])] = 0.0
mujoco.mj_forward(m, d)

R = d.xmat[base].reshape(3, 3)
o = d.xpos[base]


def verts(body, frame=False):
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


for body, sign in (("right_link7", -1), ("right_link8", +1)):
    p = verts(body) * 1000
    near = p[np.abs(p[:, 1]) < 0.5]          # the plane the jaw closes on
    print(f"\n{body}: {len(p)} verts, {len(near)} on the closing plane")
    if not len(near):
        near = p[np.argsort(np.abs(p[:, 1]))[:200]]
    print(f"  x [{near[:,0].min():7.2f}, {near[:,0].max():7.2f}] mm")
    print(f"  z [{near[:,2].min():7.2f}, {near[:,2].max():7.2f}] mm")
    # a face needs a plateau in z, not one corner
    zs, counts = np.unique(np.round(near[:, 2] / 2) * 2, return_counts=True)
    print("  z histogram (mm:count):",
          " ".join(f"{z:.0f}:{c}" for z, c in zip(zs, counts) if c > 2)[:300])
    bid = m.body(body).id
    Rb = d.xmat[bid].reshape(3, 3)
    local = (near @ R.T + o - d.xpos[bid]) @ Rb
    print(f"  in the {body} frame: x [{local[:,0].min():.4f}, {local[:,0].max():.4f}] "
          f"y [{local[:,1].min():.4f}, {local[:,1].max():.4f}] "
          f"z [{local[:,2].min():.4f}, {local[:,2].max():.4f}]")
    print(f"  centre {np.round(local.mean(axis=0), 4)}")
