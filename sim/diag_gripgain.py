"""Which gripper stiffness reproduces the recorded stall?

On the hardware the gripper never reaches its commanded opening while it holds
the cube: the command says 0.0293, the joint reads 0.0334, and the motor keeps
about 1.0 of effort. That 4.1 mm of residual error is the whole grip force. A
stiff position actuator in the sim reaches the command, the error goes to zero
and the cube is squeezed straight back out.

So the finger gain is swept and each value is scored on the two things the
recording shows: the achieved opening during the carry, and whether the cube
ends up on the yellow mat.

    python diag_gripgain.py
"""
import glob

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
LEFT_START = np.asarray([0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222,
                         0.2780382633, 0.0])
TARGET_STALL = 0.0334     # recorded achieved opening during the carry
CARRY = range(270, 341)
np.set_printoptions(precision=4, suppress=True)

df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)


def replay(kp, kv):
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    for side in ("left", "right"):
        for i in (7, 8):
            # position actuator: force = kp*ctrl - kp*q - kv*qdot, so the gain
            # and the linear bias term are the pair to change; biasprm[2] stays
            # the velocity damping.
            m.actuator_gainprm[m.actuator(f"{side}_act_{i}").id, 0] = kp
            m.actuator_biasprm[m.actuator(f"{side}_act_{i}").id, 1] = -kp
    for i in range(1, 7):
        d.qpos[int(np.ravel(m.joint(f"left_joint{i}").qposadr)[0])] = (
            SIGN[i] * LEFT_START[i - 1])
        d.ctrl[m.actuator(f"left_act_{i}").id] = SIGN[i] * LEFT_START[i - 1]
    for k in (7, 8):
        d.qpos[int(np.ravel(m.joint(f"left_joint{k}").qposadr)[0])] = 0.0
    mujoco.mj_forward(m, d)
    cube = m.body("cube").id
    j7 = int(np.ravel(m.joint("right_joint7").qposadr)[0])
    j8 = int(np.ravel(m.joint("right_joint8").qposadr)[0])
    start = d.xpos[cube].copy()
    stalls, lifted = [], 0.0
    for f in range(len(actions)):
        for i in range(1, 7):
            d.ctrl[m.actuator(f"right_act_{i}").id] = (
                SIGN[i] * float(actions[f][6 + i]))
        g = float(np.clip(actions[f][13], 0.0, 0.07))
        d.ctrl[m.actuator("right_act_7").id] = g / 2
        d.ctrl[m.actuator("right_act_8").id] = -g / 2
        for _ in range(17):
            mujoco.mj_step(m, d)
        if f in CARRY:
            stalls.append(float(d.qpos[j7] - d.qpos[j8]))
        lifted = max(lifted, float(d.xpos[cube][2]) - start[2])
    end = d.xpos[cube].copy()
    return (float(np.mean(stalls)), float(100 * (end[2] - start[2])),
            100 * float(np.linalg.norm((end - start)[:2])), bool(end[1] > 0.05),
            float(100 * lifted))


print(f"recorded stall target {TARGET_STALL:.4f}")
print(f"{'kp':>7} {'stall':>8} {'d_stall':>8} {'lift':>7} {'travel':>7} "
      f"{'on yellow':>9} {'max lift':>8}")
for kp in (3000, 1000, 500, 300, 250, 200, 150, 100, 60, 30):
    stall, end_z, travel, yellow, max_lift = replay(kp, 60)
    print(f"{kp:7.0f} {stall:8.4f} {stall - TARGET_STALL:+8.4f} {end_z:+7.2f} "
          f"{travel:7.1f} {str(yellow):>9} {max_lift:8.2f}")
