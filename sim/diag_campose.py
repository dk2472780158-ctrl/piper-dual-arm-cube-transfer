import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('/home/databall/VLA/cube_mujoco/sim/scene.xml')
d = mujoco.MjData(m)
START = {'left':[0.0492107794,0,0,-0.2287141085,0.2685223222,0.2780382633,0],
         'right':[-0.0443961099,0,0,0.0536503866,0.3397218585,-0.0396250561,0]}
for side,pose in START.items():
    for i in range(1,7):
        j=m.joint(side+'_joint'+str(i)); d.qpos[int(np.ravel(j.qposadr)[0])]=pose[i-1]
mujoco.mj_forward(m,d)
for name in ('right','left','middle'):
    cid=m.camera(name).id
    R=d.cam_xmat[cid].reshape(3,3)
    print(name,'pos',np.round(d.cam_xpos[cid],3),'look',np.round(-R[2],3),'up',np.round(R[1],3))
b=m.body('right_gripper_base')
print('right gripper axes', np.round(d.xmat[b.id].reshape(3,3)[:, :].T,3))
