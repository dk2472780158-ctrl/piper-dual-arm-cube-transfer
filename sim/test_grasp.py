"""Does the arm actually pick the cube up at the solved layout?

Drives the right arm with the recorded joint trajectory (start -> grasp ->
close -> lift) and reports whether the cube rises. The arm base height is a
free parameter: the recorded pick/place pair cannot pin it down exactly, so it
is swept and the height that grasps cleanly wins.

Writes nothing; prints one line per candidate.
"""
import glob

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
START = np.asarray([-0.0443961099, 0.0, 0.0, 0.0536503866, 0.3397218585,
                    -0.0396250561, 0.0])
LEFT_START = np.asarray([0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222,
                         0.2780382633, 0.0])
BASE_XY = (-0.3765, -0.3020)
YAW = 0.026180
CUBE_XY = (0.0, -0.1110)
GRASP, LIFT = 270, 330
TABLE = 0.72
CUBE_HALF = 0.0225

df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
mount = m.body("right_mount").id
cube_body = m.body("cube").id
free_adr = int(m.joint("cube_free").qposadr[0])


def set_ctrl(pose7):
    for i in range(1, 7):
        d.ctrl[m.actuator(f"right_act_{i}").id] = SIGN[i] * float(pose7[i - 1])
    g = float(np.clip(pose7[6], 0.0, 0.07))
    d.ctrl[m.actuator("right_act_7").id] = g / 2
    d.ctrl[m.actuator("right_act_8").id] = -g / 2


def run(base_z):
    mujoco.mj_resetData(m, d)
    m.body_pos[mount][2] = base_z
    # the cube keeps the spawn from scene.xml
    # hold both arms at the recorded start (the left arm has its own start)
    for side in ("left", "right"):
        pose = START if side == "right" else LEFT_START
        for i in range(1, 7):
            j = m.joint(f"{side}_joint{i}")
            d.qpos[int(np.ravel(j.qposadr)[0])] = SIGN[i] * pose[i - 1]
        for k in (7, 8):
            d.qpos[int(np.ravel(m.joint(f"{side}_joint{k}").qposadr)[0])] = 0.0
        for i in range(1, 7):
            d.ctrl[m.actuator(f"{side}_act_{i}").id] = SIGN[i] * pose[i - 1]
        d.ctrl[m.actuator(f"{side}_act_7").id] = 0.0
        d.ctrl[m.actuator(f"{side}_act_8").id] = 0.0
    mujoco.mj_forward(m, d)
    c0 = d.xpos[cube_body].copy()

    waypoints = [(START, actions[GRASP][7:14], 90),      # reach
                 (actions[GRASP][7:14], actions[GRASP][7:14], 15),  # settle
                 (actions[GRASP][7:14], actions[LIFT][7:14], 75)]   # lift
    for src, dst, steps in waypoints:
        src, dst = np.asarray(src, float), np.asarray(dst, float)
        for k in range(1, steps + 1):
            set_ctrl(src + (dst - src) * k / steps)
            for _ in range(17):
                mujoco.mj_step(m, d)
    drop = float(np.linalg.norm(d.xpos[cube_body] - c0))
    return drop, c0, d.xpos[cube_body].copy()


def detail(base_z):
    """Same drive, with a report at the moment the gripper has just closed."""
    mujoco.mj_resetData(m, d)
    m.body_pos[mount][2] = base_z
    # the cube keeps the spawn from scene.xml
    for side in ("left", "right"):
        pose = START if side == "right" else LEFT_START
        for i in range(1, 7):
            d.qpos[int(np.ravel(m.joint(f"{side}_joint{i}").qposadr)[0])] = (
                SIGN[i] * pose[i - 1])
            d.ctrl[m.actuator(f"{side}_act_{i}").id] = SIGN[i] * pose[i - 1]
        d.ctrl[m.actuator(f"{side}_act_7").id] = 0.0
        d.ctrl[m.actuator(f"{side}_act_8").id] = 0.0
    mujoco.mj_forward(m, d)
    src, dst = START, np.asarray(actions[GRASP][7:14], float)
    for k in range(1, 91):
        set_ctrl(src + (dst - src) * k / 90)
        for _ in range(17):
            mujoco.mj_step(m, d)
    for _ in range(30):
        mujoco.mj_step(m, d)
    err = [SIGN[i] * float(d.qpos[int(np.ravel(m.joint(f"right_joint{i}").qposadr)[0])])
           - float(actions[GRASP][6 + i]) for i in range(1, 7)]
    grip = float(d.qpos[int(np.ravel(m.joint("right_joint7").qposadr)[0])]) - \
        float(d.qpos[int(np.ravel(m.joint("right_joint8").qposadr)[0])])
    print(f"base_z {base_z:.4f}: joint err {np.round(err, 3).tolist()} "
          f"grip {grip:.4f} (commanded {actions[GRASP][13]:.4f})")
    print("  cube     ", np.round(d.xpos[cube_body], 3))
    for body in ("right_link7", "right_link8"):
        bid = m.body(body).id
        print(f"  {body:11s}", np.round(d.xpos[bid], 3))
    print("  ncon", d.ncon)


print("cube spawn", [CUBE_XY[0], CUBE_XY[1], round(TABLE + 0.003 + CUBE_HALF, 4)])
detail(0.7120)
for base_z in np.arange(0.700, 0.7301, 0.002):
    drop, c0, c1 = run(base_z)
    print(f"base_z {base_z:.3f}  cube moved {drop * 100:5.2f} cm  "
          f"end {np.round(c1, 3)}")
