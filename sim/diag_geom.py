import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("/home/databall/VLA/cube_mujoco/sim/scene.xml")
d = mujoco.MjData(m)
START = {"left":[0.0492107794,0,0,-0.2287141085,0.2685223222,0.2780382633,0],
         "right":[-0.0443961099,0,0,0.0536503866,0.3397218585,-0.0396250561,0]}
for side,pose in START.items():
    for i in range(1,7):
        j=m.joint(f"{side}_joint{i}"); d.qpos[int(np.ravel(j.qposadr)[0])]=pose[i-1]
    for i in (7,8):
        j=m.joint(f"{side}_joint{i}"); d.qpos[int(np.ravel(j.qposadr)[0])]=0.0
mujoco.mj_forward(m,d)
np.set_printoptions(precision=4, suppress=True)
for name in ["left_gripper_base","left_link7","left_link8","right_gripper_base","right_link7","right_link8","cube"]:
    b=m.body(name); bid=b.id
    print(name, "pos", np.round(d.xpos[bid],4))
    print("   xaxis", np.round(d.xmat[bid][0:3],3), "yaxis", np.round(d.xmat[bid][3:6],3), "zaxis", np.round(d.xmat[bid][6:9],3))
