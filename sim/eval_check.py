"""Report where the cube ended up in the newest policy run.

The sim is deterministic, so replaying the recorded action column reproduces
the episode. Loads the middle-camera video as well so a few frames can be
inspected.
"""
import glob
import os

import numpy as np
import pandas as pd
import mujoco
import imageio.v2 as imageio

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
LEFT_START = np.asarray([0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222,
                         0.2780382633, 0.0])
np.set_printoptions(precision=4, suppress=True)

run = sorted(glob.glob("/home/databall/.cache/huggingface/lerobot/local/"
                       "eval_sim_cube_r2l_020000_*"))[-1]
print("run", run)
files = sorted(glob.glob(f"{run}/data/chunk-*/file-*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files]).reset_index(drop=True)
actions = np.stack(df["action"].values)
print("frames", len(actions))

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
cube = m.body("cube").id


def drive(pose7):
    for i in range(1, 7):
        d.ctrl[m.actuator(f"right_act_{i}").id] = SIGN[i] * float(pose7[i - 1])
    g = float(np.clip(pose7[6], 0.0, 0.07))
    d.ctrl[m.actuator("right_act_7").id] = g / 2
    d.ctrl[m.actuator("right_act_8").id] = -g / 2


mujoco.mj_resetData(m, d)
for i in range(1, 7):
    d.qpos[int(np.ravel(m.joint(f"left_joint{i}").qposadr)[0])] = (
        SIGN[i] * LEFT_START[i - 1])
    d.ctrl[m.actuator(f"left_act_{i}").id] = SIGN[i] * LEFT_START[i - 1]
for k in (7, 8):
    d.qpos[int(np.ravel(m.joint(f"left_joint{k}").qposadr)[0])] = 0.0
mujoco.mj_forward(m, d)
start = d.xpos[cube].copy()
print("spawn", np.round(start, 4))
for f in range(len(actions)):
    drive(actions[f][7:14])
    for _ in range(17):
        mujoco.mj_step(m, d)
    if f % 40 == 0:
        print(f"  f{f:3d} cube {np.round(d.xpos[cube], 3)}")

end = d.xpos[cube].copy()
print("end  ", np.round(end, 4))
print("displacement %.1f cm, lift %.1f cm" % (
    100 * np.linalg.norm(end - start), 100 * (end[2] - start[2])))
print("reached yellow mat (y>0.06):", bool(end[1] > 0.06))

vids = glob.glob(f"{run}/videos/**/middle/*.mp4", recursive=True)
if vids:
    reader = imageio.get_reader(vids[0])
    n = reader.count_frames()
    print("video", os.path.basename(vids[0]), n, "frames")
    os.makedirs("/tmp/evalframes", exist_ok=True)
    for idx in (0, n // 4, n // 2, (3 * n) // 4, n - 1):
        imageio.imwrite(f"/tmp/evalframes/m_{idx:04d}.png", reader.get_data(idx))
    print("wrote /tmp/evalframes")
