import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("/home/databall/VLA/cube_mujoco/sim/scene.xml")
d = mujoco.MjData(m)
START = [0.0492107794,0,0,-0.2287141085,0.2685223222,0.2780382633,0,
         -0.0443961099,0,0,0.0536503866,0.3397218585,-0.0396250561,0]
for side,off in (("left",0),("right",7)):
    for i in range(1,7):
        j=m.joint(f"{side}_joint{i}"); d.qpos[int(np.ravel(j.qposadr)[0])]=START[off+i-1]
        d.ctrl[m.actuator(f"{side}_act_{i}").id]=START[off+i-1]
for side in ("left","right"):
    for i in (7,8):
        d.ctrl[m.actuator(f"{side}_act_{i}").id]=0.0
mujoco.mj_forward(m,d)
print("ncon", d.ncon)
from collections import Counter
c=Counter()
for k in range(d.ncon):
    gc=d.contact[k]
    b1=m.geom_bodyid[gc.geom1]; b2=m.geom_bodyid[gc.geom2]
    c[tuple(sorted((mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b1) or "world",
                    mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b2) or "world")))]+=1
for k,v in c.most_common(20): print(v,k)
print("--- body masses ---")
for b in range(m.nbody):
    print(b, mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b), round(float(m.body_mass[b]),4))
