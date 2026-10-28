import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("/home/databall/VLA/cube_mujoco/sim/scene.xml")
d = mujoco.MjData(m)
START = {"left":[0.0492107794,0,0,-0.2287141085,0.2685223222,0.2780382633,0],
         "right":[-0.0443961099,0,0,0.0536503866,0.3397218585,-0.0396250561,0]}
for side,pose in START.items():
    for i in range(1,7):
        j=m.joint(f"{side}_joint{i}"); d.qpos[int(np.ravel(j.qposadr)[0])]=pose[i-1]
mujoco.mj_forward(m,d)
r = mujoco.Renderer(m, 480, 640)
r.enable_segmentation_rendering()
for cam in ("right","left","middle"):
    r.update_scene(d, camera=cam)
    seg = r.render()  # (h,w,2): [object id, object type]
    print(cam, "seg dtype", seg.dtype, seg.shape)
    for (y,x,tag) in ((400,100,"bottom-left"),(400,320,"bottom-mid"),(240,320,"centre"),(120,320,"top-mid")):
        oid, otype = int(seg[y,x,0]), int(seg[y,x,1])
        name = None
        if otype == mujoco.mjtObj.mjOBJ_GEOM:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, oid)
        elif otype == mujoco.mjtObj.mjOBJ_BODY:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, oid)
        print(f"   {tag:12s} type={otype} id={oid} name={name}")
