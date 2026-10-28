import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("/home/databall/VLA/cube_mujoco/sim/scene.xml")
d = mujoco.MjData(m)
LEFT = [0.0283906957,0.00927215312,-0.0177944968,-0.0329544165,0.283687453,-0.0542393676,1.408e-05]
RIGHT = [0.521147583,1.84114562,-1.28865731,0.0781050022,0.862001224,-0.0330817683,0.036436876]
for side, pose in (("left",LEFT),("right",RIGHT)):
    for i in range(1,7):
        j=m.joint(f"{side}_joint{i}"); d.qpos[int(np.ravel(j.qposadr)[0])]=pose[i-1]
        d.ctrl[m.actuator(f"{side}_act_{i}").id]=pose[i-1]
    g=pose[6]
    d.ctrl[m.actuator(f"{side}_act_7").id]=g/2; d.ctrl[m.actuator(f"{side}_act_8").id]=-g/2
mujoco.mj_forward(m,d)
print("cube pos", d.qpos[0:3].round(4))
def fingers():
    a=float(d.qpos[int(np.ravel(m.joint("right_joint7").qposadr)[0])])
    b=float(d.qpos[int(np.ravel(m.joint("right_joint8").qposadr)[0])])
    return a,b,a-b
print("fingers before settle", [round(v,5) for v in fingers()])
for _ in range(60): mujoco.mj_step(m,d)
print("fingers after 60 steps", [round(v,5) for v in fingers()], "ncon", d.ncon)
for k in range(d.ncon):
    c=d.contact[k]; b1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom1]); b2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom2])
    print("  contact", b1, b2, "dist", round(float(c.dist),4))
g=0.0364
for step in range(1,120):
    g=max(0.0, g-0.0003333)
    d.ctrl[m.actuator("right_act_7").id]=g/2; d.ctrl[m.actuator("right_act_8").id]=-g/2
    for _ in range(17): mujoco.mj_step(m,d)
    if step%20==0: print(f"step {step} cmd={g:.4f} fingers={[round(v,5) for v in fingers()]} err={fingers()[2]-g:+.5f}")
print("cube pos end", d.qpos[0:3].round(4))
