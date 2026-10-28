import numpy as np
import mujoco

m = mujoco.MjModel.from_xml_path(
    "/home/databall/VLA/cube_mujoco/sim/scene.xml")
d = mujoco.MjData(m)
START = {
    "left": [0.0492107794, 0, 0, -0.2287141085, 0.2685223222, 0.2780382633, 0],
    "right": [-0.0443961099, 0, 0, 0.0536503866, 0.3397218585, -0.0396250561, 0],
}
for side, pose in START.items():
    for i in range(1, 7):
        j = m.joint(f"{side}_joint{i}")
        d.qpos[int(np.ravel(j.qposadr)[0])] = pose[i - 1]
mujoco.mj_forward(m, d)

for name in ("right", "left"):
    cid = m.camera(name).id
    bid = int(m.cam_bodyid[cid])
    print(name, "cam_bodyid", bid, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid))
    M = d.xmat[bid].reshape(3, 3)
    print("  xmat rows    ", np.round(M, 3).tolist())
    v = m.cam_pos[cid]
    print("  rows-as-axes ", np.round(d.xpos[bid] + M.T @ v, 4))
    print("  direct M@v   ", np.round(d.xpos[bid] + M @ v, 4))
    print("  cam_xpos     ", np.round(d.cam_xpos[cid], 4))
    print("  body xpos    ", np.round(d.xpos[bid], 4))
