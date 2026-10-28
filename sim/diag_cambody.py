import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('/home/databall/VLA/cube_mujoco/sim/scene.xml')
d = mujoco.MjData(m)
for name in ('right','left','middle'):
    cid=m.camera(name).id
    bid=m.cam_bodyid[cid]
    print(name,'body',mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bid),
          'cam_pos_local',np.round(m.cam_pos[cid],4),
          'xyaxes',np.round(m.cam_quat[cid],4))
print('body ids:')
for b in range(m.nbody):
    print(' ',b,mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b))
