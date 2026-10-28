"""At what opening do the finger pads first touch the cube?

The recorded hold is a *residual*: the hardware is commanded to 0.0293 and
reads 0.0334, and that 4.1 mm of error is the grip. If the sim's pads do not
touch the cube until the joint is well below 0.0334, then commanding 0.0293
puts no load on the cube at all and the fingers settle without any grip.

The arm is held at the recorded grasp pose and the gripper is stepped shut;
the joint value at which contact appears is printed.

    python diag_aperture.py
"""
import glob

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
GRASP = 255
np.set_printoptions(precision=4, suppress=True)

df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
cube = m.body("cube").id
j7 = int(np.ravel(m.joint("right_joint7").qposadr)[0])
j8 = int(np.ravel(m.joint("right_joint8").qposadr)[0])
pose = np.asarray(actions[GRASP][7:14], float)

for i in range(1, 7):
    j = int(np.ravel(m.joint(f"right_joint{i}").qposadr)[0])
    d.qpos[j] = SIGN[i] * pose[i - 1]
    d.ctrl[m.actuator(f"right_act_{i}").id] = SIGN[i] * pose[i - 1]


def touches():
    for k in range(d.ncon):
        c = d.contact[k]
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                               m.geom_bodyid[c.geom1]) or "world"
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                               m.geom_bodyid[c.geom2]) or "world"
        if "cube" in (n1, n2) and (n1, n2) != ("cube", "cube"):
            return True
    return False


print(f"grasp pose from frame {GRASP}, cube at {np.round(d.xpos[cube], 4)}")
print(f"{'cmd':>7} {'open':>7} {'cube xyz':>22} {'grip':>7}  touch")
for cmd in np.arange(0.070, 0.0199, -0.002):
    d.qpos[j7] = cmd / 2
    d.qpos[j8] = -cmd / 2
    d.ctrl[m.actuator("right_act_7").id] = cmd / 2
    d.ctrl[m.actuator("right_act_8").id] = -cmd / 2
    mujoco.mj_forward(m, d)
    print(f"{cmd:7.3f} {d.qpos[j7] - d.qpos[j8]:7.4f} "
          f"{str(np.round(d.xpos[cube], 4)):>22} {d.qpos[j7] - d.qpos[j8]:7.4f}  "
          f"{touches()}")

# then let the physics run at a few commands to see the settled state
print("\nsettled, 1.5 s at each command:")
for cmd in (0.040, 0.035, 0.030, 0.025, 0.020, 0.0):
    mujoco.mj_resetData(m, d)
    for i in range(1, 7):
        j = int(np.ravel(m.joint(f"right_joint{i}").qposadr)[0])
        d.qpos[j] = SIGN[i] * pose[i - 1]
        d.ctrl[m.actuator(f"right_act_{i}").id] = SIGN[i] * pose[i - 1]
    d.qpos[j7] = 0.035
    d.qpos[j8] = -0.035
    d.ctrl[m.actuator("right_act_7").id] = cmd / 2
    d.ctrl[m.actuator("right_act_8").id] = -cmd / 2
    mujoco.mj_forward(m, d)
    cube0 = d.xpos[cube].copy()
    for _ in range(750):
        mujoco.mj_step(m, d)
    print(f"  cmd {cmd:5.3f}  open {d.qpos[j7] - d.qpos[j8]:7.4f}  "
          f"cube {np.round(d.xpos[cube], 4)}  "
          f"moved {1000 * np.linalg.norm(d.xpos[cube] - cube0):+6.1f} mm  "
          f"touch {touches()}")
