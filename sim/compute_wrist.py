"""Derive a fixed wrist-camera pose (gripper_base frame) from a world-space intent."""
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
cube = d.xpos[m.body("cube").id]
UP = np.array([0.0,0.0,1.0])
for side in ("left","right"):
    bid = m.body(f"{side}_gripper_base").id
    pos = d.xpos[bid].copy()
    R = d.xmat[bid].reshape(3,3).T        # columns are the body axes
    approach = R[:,2]                      # local +z in world
    tip = d.xpos[m.body(f"{side}_link7").id]
    cam_world = pos + UP*0.06 + approach*0.02
    # look 0.22 m further along the approach, 0.05 m above the table
    target = cam_world + approach*0.22 - UP*(cam_world[2] - (0.72 + 0.05))
    target[2] = 0.72 + 0.05
    dirw = target - cam_world; dirw /= np.linalg.norm(dirw)
    upw = UP - dirw*np.dot(UP,dirw); upw /= np.linalg.norm(upw)
    loc_pos = R.T @ (cam_world - pos)
    loc_dir = R.T @ dirw
    loc_up  = R.T @ upw
    print(side, "base", np.round(pos,3), "tip", np.round(tip,3))
    print("  pos_local", np.round(loc_pos,4), "dir_local", np.round(loc_dir,4),
          "up_local", np.round(loc_up,4), "tilt_deg", round(float(np.degrees(np.arcsin(-dirw[2]))),1))
