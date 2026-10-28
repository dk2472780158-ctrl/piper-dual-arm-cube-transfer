"""Lowest world z of every right-arm collision mesh at the recorded end pose."""
import numpy as np
import mujoco

m = mujoco.MjModel.from_xml_path("/home/databall/VLA/cube_mujoco/sim/scene.xml")
d = mujoco.MjData(m)
CURRENT_R = [5.19811826e-01, 1.95424033e+00, -1.44235626e+00, 1.46603260e-02,
             8.55827040e-01, 7.34690199e-03, 3.01020937e-02]
CURRENT_L = [6.46667390e-03, 2.31143216e-03, -4.45598130e-03, -1.31030630e-02,
             2.87655252e-01, -6.03176970e-02, 1.19105878e-05]
for side, pose in (("left", CURRENT_L), ("right", CURRENT_R)):
    for i in range(1, 7):
        j = m.joint(f"{side}_joint{i}")
        d.qpos[int(np.ravel(j.qposadr)[0])] = pose[i - 1]
    for i in (7, 8):
        j = m.joint(f"{side}_joint{i}")
        d.qpos[int(np.ravel(j.qposadr)[0])] = (pose[6] / 2) * (1 if i == 7 else -1)
mujoco.mj_forward(m, d)
print("table top z = 0.72")
for side in ("left", "right"):
    for link in ("link1", "link2", "link3", "link4", "link5", "link6",
                 "gripper_base", "link7", "link8"):
        bid = m.body(f"{side}_{link}").id
        lowest = 1e9
        for g in range(m.body_geomadr[bid], m.body_geomadr[bid] + m.body_geomnum[bid]):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if m.geom_contype[g] == 0:
                continue
            mh = m.geom_dataid[g]
            a = m.mesh_vertadr[mh]
            n = m.mesh_vertnum[mh]
            v = m.mesh_vert[a:a + n]
            R = d.xmat[bid].reshape(3, 3)
            world = v @ R.T + d.xpos[bid]
            lowest = min(lowest, float(world[:, 2].min()))
        if lowest < 1e8:
            print(f"  {side}_{link:13s} lowest collision z = {lowest:.4f}")
