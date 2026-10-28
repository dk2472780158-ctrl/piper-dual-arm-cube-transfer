"""Trace one policy run: where did each gripper go, and how far from the cube.

Also compares the run's action column with the recorded demonstration's so a
policy drift can be told apart from a replay bug.
"""
import glob

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
DEMO = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
LEFT_START = np.asarray([0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222,
                         0.2780382633, 0.0])
np.set_printoptions(precision=4, suppress=True)

run = sorted(glob.glob("/home/databall/.cache/huggingface/lerobot/local/"
                       "eval_sim_cube_r2l_020000_*"))[-1]
files = sorted(glob.glob(f"{run}/data/chunk-*/file-*.parquet"))
act = np.stack(pd.concat([pd.read_parquet(f) for f in files])
               .reset_index(drop=True)["action"].values)

demo = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(DEMO))])
demo = demo[demo.episode_index == 0].reset_index(drop=True)
demo_act = np.stack(demo["action"].values)
print("run frames", len(act), "demo frames", len(demo_act))
for f in (0, 20, 60, 100, 160):
    print(f"f{f:3d} run  ", np.round(act[f], 3))
    print(f"     demo ", np.round(demo_act[min(f, len(demo_act) - 1)], 3))

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)


def drive(pose7, side):
    for i in range(1, 7):
        d.ctrl[m.actuator(f"{side}_act_{i}").id] = SIGN[i] * float(pose7[i - 1])
    g = float(np.clip(pose7[6], 0.0, 0.07))
    d.ctrl[m.actuator(f"{side}_act_7").id] = g / 2
    d.ctrl[m.actuator(f"{side}_act_8").id] = -g / 2


mujoco.mj_resetData(m, d)
for i in range(1, 7):
    v = SIGN[i] * LEFT_START[i - 1]
    d.qpos[int(np.ravel(m.joint(f"left_joint{i}").qposadr)[0])] = v
    d.ctrl[m.actuator(f"left_act_{i}").id] = v
mujoco.mj_forward(m, d)
cube = m.body("cube").id
print("\nframe   right_tcp              left_tcp               cube        gripper")
for f in range(len(act)):
    drive(act[f][:7], "left")
    drive(act[f][7:], "right")
    for _ in range(17):
        mujoco.mj_step(m, d)
    if f % 20 == 0 or f == len(act) - 1:
        rt = d.xpos[m.body("right_link7").id]
        lt = d.xpos[m.body("left_link7").id]
        print(f"{f:5d}  {np.round(rt, 3)}  {np.round(lt, 3)}  "
              f"{np.round(d.xpos[cube], 3)}  g={act[f][13]:.3f} "
              f"|r-c|={np.linalg.norm(rt - d.xpos[cube]):.3f}")
