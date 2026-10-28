"""Where in the carry does the cube leave the gripper, and why?

Replays the recorded demonstration and prints, every few frames, the cube
position, the finger opening compared with its command, and the normal force
the fingers put on the cube. The normal force is the number that matters: the
cube can only stay up if friction times that force beats its weight and the
acceleration of the swing.

    python diag_carry.py            # production finger gain
    python diag_carry.py 100        # a softer finger
"""
import glob
import sys

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
LEFT_START = np.asarray([0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222,
                         0.2780382633, 0.0])
KP = float(sys.argv[1]) if len(sys.argv) > 1 else 3000.0
EVERY = 5
np.set_printoptions(precision=4, suppress=True)

df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
for side in ("left", "right"):
    for i in (7, 8):
        a = m.actuator(f"{side}_act_{i}").id
        m.actuator_gainprm[a, 0] = KP
        m.actuator_biasprm[a, 1] = -KP

cube = m.body("cube").id
j7 = int(np.ravel(m.joint("right_joint7").qposadr)[0])
j8 = int(np.ravel(m.joint("right_joint8").qposadr)[0])

for i in range(1, 7):
    d.qpos[int(np.ravel(m.joint(f"left_joint{i}").qposadr)[0])] = (
        SIGN[i] * LEFT_START[i - 1])
    d.ctrl[m.actuator(f"left_act_{i}").id] = SIGN[i] * LEFT_START[i - 1]
for k in (7, 8):
    d.qpos[int(np.ravel(m.joint(f"left_joint{k}").qposadr)[0])] = 0.0
mujoco.mj_forward(m, d)
start = d.xpos[cube].copy()


def cube_contacts():
    """Normal force and the other body, for every contact touching the cube."""
    out = []
    for k in range(d.ncon):
        c = d.contact[k]
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1) or "world"
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2) or "world"
        if "cube" not in (n1, n2) or (n1, n2) == ("cube", "cube"):
            continue
        other = n2 if n1 == "cube" else n1
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, k, f)
        out.append((other, f[0]))
    return out


print(f"finger gain {KP:.0f}")
print(f"{'f':>4} {'cube xyz':>22} {'cmd':>7} {'open':>7} {'lift':>7}  contacts")
for f in range(len(actions)):
    for i in range(1, 7):
        d.ctrl[m.actuator(f"right_act_{i}").id] = SIGN[i] * float(actions[f][6 + i])
    g = float(np.clip(actions[f][13], 0.0, 0.07))
    d.ctrl[m.actuator("right_act_7").id] = g / 2
    d.ctrl[m.actuator("right_act_8").id] = -g / 2
    for _ in range(17):
        mujoco.mj_step(m, d)
    if f < 230 or f % EVERY:
        continue
    opening = float(d.qpos[j7] - d.qpos[j8])
    hits = cube_contacts()
    total = sum(abs(n) for _, n in hits if n < 0)
    desc = " ".join(f"{n}:{v:+.2f}" for n, v in hits)
    print(f"{f:4d} {str(np.round(d.xpos[cube], 3)):>22} {g:7.4f} {opening:7.4f} "
          f"{100 * (d.xpos[cube][2] - start[2]):+7.2f}  {desc}")

end = d.xpos[cube].copy()
print("end", np.round(end, 4), "travel %.1f cm" %
      (100 * np.linalg.norm((end - start)[:2])))
