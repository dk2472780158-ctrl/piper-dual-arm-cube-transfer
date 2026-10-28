"""Where did the cube end up in the newest simulated policy run?

The sim is deterministic, so replaying the recorded action column -- both arms,
the same start pose and the same number of physics steps per frame -- reproduces
the episode. The cube is not recorded in the dataset, so this replay is the only
way to see what happened to it.
"""
import glob
import os

import numpy as np
import pandas as pd
import mujoco
import imageio.v2 as imageio

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
START = np.asarray([0.0093, 0.0049, -0.0024, -0.0582, 0.3185, 0.0973, 0.0003,
                    0.0631, 0.0048, -0.0027, -0.0766, 0.3512, 0.0202, 0.0])
np.set_printoptions(precision=4, suppress=True)

run = sorted(glob.glob("/home/databall/.cache/huggingface/lerobot/local/"
                       "eval_sim_cube_r2l_020000_*"))[-1]
print("run", os.path.basename(run))
files = sorted(glob.glob(f"{run}/data/chunk-*/file-*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files]).reset_index(drop=True)
actions = np.stack(df["action"].values)
print("frames", len(actions))

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
cube = m.body("cube").id


def drive(pose14):
    for side, base in (("left", 0), ("right", 7)):
        pose = pose14[base:base + 7]
        for i in range(1, 7):
            d.ctrl[m.actuator(f"{side}_act_{i}").id] = SIGN[i] * float(pose[i - 1])
        grip = max(0.0, min(0.07, float(pose[6])))
        d.ctrl[m.actuator(f"{side}_act_7").id] = grip / 2
        d.ctrl[m.actuator(f"{side}_act_8").id] = -grip / 2


mujoco.mj_resetData(m, d)
for side, base in (("left", 0), ("right", 7)):
    for i in range(1, 7):
        value = SIGN[i] * float(START[base + i - 1])
        d.qpos[int(np.ravel(m.joint(f"{side}_joint{i}").qposadr)[0])] = value
        d.ctrl[m.actuator(f"{side}_act_{i}").id] = value
    for k in (7, 8):
        d.qpos[int(np.ravel(m.joint(f"{side}_joint{k}").qposadr)[0])] = 0.0
mujoco.mj_forward(m, d)

start = d.xpos[cube].copy()
print("spawn", np.round(start, 4))
for f in range(len(actions)):
    drive(actions[f])
    for _ in range(17):
        mujoco.mj_step(m, d)
    if f % 50 == 0:
        print(f"  f{f:3d} cube {np.round(d.xpos[cube], 3)} "
              f"right j2 {SIGN[2] * d.qpos[int(np.ravel(m.joint('right_joint2').qposadr)[0])]:+.2f}")
end = d.xpos[cube].copy()
print("end  ", np.round(end, 4))
print("displacement %.1f cm, lift %.1f cm" % (
    100 * np.linalg.norm(end[:2] - start[:2]), 100 * (end[2] - start[2])))
print("reached yellow mat (y > 0.05):", bool(end[1] > 0.05))

for path in glob.glob(f"{run}/videos/observation.images.middle/*/*.mp4"):
    reader = imageio.get_reader(path)
    total = reader.count_frames()
    print("video", os.path.basename(path), total, "frames")
    os.makedirs("/tmp/evalframes", exist_ok=True)
    for index in sorted({0, total // 4, total // 2, 3 * total // 4, total - 1}):
        imageio.imwrite(f"/tmp/evalframes/m_{index:04d}.png", reader.get_data(index))
    print("wrote /tmp/evalframes")
