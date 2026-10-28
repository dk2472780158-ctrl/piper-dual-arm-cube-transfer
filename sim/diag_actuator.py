import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("/home/databall/VLA/cube_mujoco/sim/scene.xml")
d = mujoco.MjData(m)
for a in range(m.nu):
    jid = m.actuator_trnid[a,0]
    print(f"act {a:2d} {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a):12s} -> joint {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid):12s} "
          f"kp={m.actuator_gainprm[a,0]:.1f} forcerange={m.actuator_forcerange[a]} ctrlrange={m.actuator_ctrlrange[a]} "
          f"qposadr={m.jnt_qposadr[jid]} dofadr={m.jnt_dofadr[jid]} damping={m.dof_damping[m.jnt_dofadr[jid]]:.2f} armature={m.dof_armature[m.jnt_dofadr[jid]]:.3f}")
print("total mass", m.body_subtreemass[1])
j1 = m.joint("left_joint1")
adr = int(np.ravel(j1.qposadr)[0])
print("left_joint1 qpos range", m.jnt_range[j1.id], "limited", m.jnt_limited[j1.id])
# command +0.2 on every left arm joint and step
a1 = m.actuator("left_act_1").id
print("ctrl before", d.ctrl[a1], "qpos", d.qpos[adr])
d.ctrl[a1] = d.qpos[adr] + 0.2
for k in range(1, 501):
    mujoco.mj_step(m, d)
    if k in (1, 5, 10, 20, 50, 100, 200, 500):
        print(f"step {k:3d} qpos={d.qpos[adr]:+.5f} qvel={d.qvel[int(np.ravel(j1.dofadr)[0])]:+.5f} actfrc={d.qfrc_actuator[int(np.ravel(j1.dofadr)[0])]:+.3f} balance={d.actuator_force[a1]:+.3f}")
