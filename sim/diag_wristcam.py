"""Express the wrist-camera targets in the right gripper_base frame.

At the training start pose the recorded wrist image shows the two fingers at
the bottom of the frame and the purple mat with the cube ahead to the left.
Printing those points in the gripper frame gives the camera a fixed target.
"""
import numpy as np
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
START = {"left": [0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222, 0.2780382633],
         "right": [-0.0443961099, 0.0, 0.0, 0.0536503866, 0.3397218585, -0.0396250561]}
np.set_printoptions(precision=4, suppress=True)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
for side, pose in START.items():
    for i in range(1, 7):
        d.qpos[int(np.ravel(m.joint(f"{side}_joint{i}").qposadr)[0])] = SIGN[i] * pose[i - 1]
mujoco.mj_forward(m, d)

for side in ("right", "left"):
    base = m.body(f"{side}_gripper_base").id
    o = d.xpos[base]
    R = d.xmat[base].reshape(3, 3)          # rows are the world axes of the body frame
    print(side, "gripper_base world", np.round(o, 4))
    for name in ("link7", "link8"):
        p = d.xpos[m.body(f"{side}_{name}").id]
        print(f"   {name} local {np.round(R.T @ (p - o), 4)}")
    for name in ("cube", "yellow_mat", "purple_mat"):
        bid = m.body(name).id if name == "cube" else None
        p = d.xpos[bid] if bid is not None else np.array(
            [float(v) for v in m.geom(name).pos])
        if name != "cube":
            p = d.geom_xpos[m.geom(f"{name}_mat").id]
        print(f"   {name:11s} local {np.round(R.T @ (p - o), 4)} world {np.round(p, 4)}")
    print()
