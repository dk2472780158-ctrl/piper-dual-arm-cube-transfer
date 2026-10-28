"""Replay the auto-reset ramp from a recorded end pose and report per-joint lag."""
import numpy as np
import mujoco

HERE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
CURRENT = {
    "left": [6.46667390e-03, 2.31143216e-03, -4.45598130e-03, -1.31030630e-02,
             2.87655252e-01, -6.03176970e-02, 1.19105878e-05],
    "right": [5.19811826e-01, 1.95424033e+00, -1.44235626e+00, 1.46603260e-02,
              8.55827040e-01, 7.34690199e-03, 3.01020937e-02],
}
TARGET = {
    "left": CURRENT["left"],
    "right": [-4.43961099e-02, 0.0, 0.0, 5.36503866e-02, 3.39721858e-01,
              -3.96250561e-02, 0.0],
}

m = mujoco.MjModel.from_xml_path(HERE)
d = mujoco.MjData(m)
for side, pose in CURRENT.items():
    for i in range(1, 7):
        j = m.joint(f"{side}_joint{i}")
        d.qpos[int(np.ravel(j.qposadr)[0])] = pose[i - 1]
        d.ctrl[m.actuator(f"{side}_act_{i}").id] = pose[i - 1]
    g = pose[6]
    d.ctrl[m.actuator(f"{side}_act_7").id] = g / 2
    d.ctrl[m.actuator(f"{side}_act_8").id] = -g / 2
mujoco.mj_forward(m, d)
bid = m.body("right_gripper_base").id
print("right gripper_base world z", round(float(d.xpos[bid][2]), 4), "(table top 0.72)")
cube = m.body("cube").id
print("cube", np.round(d.xpos[cube], 4))

command = {s: list(p) for s, p in CURRENT.items()}
step_size = 0.8 / 30.0
grip_step = 0.01 / 30.0
for step in range(180):
    # measure
    err = {}
    for side in CURRENT:
        for i in range(1, 7):
            q = float(d.qpos[int(np.ravel(m.joint(f"{side}_joint{i}").qposadr)[0])])
            err[f"{side}{i}"] = q - command[side][i - 1]
    worst = max(err, key=lambda k: abs(err[k]))
    if step % 15 == 0 or abs(err[worst]) > 0.1:
        print(f"step {step:3d} worst {worst} err {err[worst]:+.4f} ncon {d.ncon} "
              f"gripZ {d.xpos[bid][2]:.3f}")
        for k in range(d.ncon):
            c = d.contact[k]
            b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2])
            print("      contact", b1, b2, "dist", round(float(c.dist), 4),
                  "fric", round(float(np.linalg.norm(d.efc_force[:])), 2) if False else "")
        for i in range(1, 7):
            a = m.actuator(f"right_act_{i}").id
            print(f"      act{i} force {d.actuator_force[a]:+.2f}", end="")
        print()
        if abs(err[worst]) > 0.12:
            break
    # ramp + apply
    for side in CURRENT:
        for i in range(1, 7):
            delta = TARGET[side][i - 1] - command[side][i - 1]
            command[side][i - 1] += float(np.clip(delta, -step_size, step_size))
            d.ctrl[m.actuator(f"{side}_act_{i}").id] = command[side][i - 1]
        delta = TARGET[side][6] - command[side][6]
        command[side][6] += float(np.clip(delta, -grip_step, grip_step))
        d.ctrl[m.actuator(f"{side}_act_7").id] = command[side][6] / 2
        d.ctrl[m.actuator(f"{side}_act_8").id] = -command[side][6] / 2
    for _ in range(17):
        mujoco.mj_step(m, d)
