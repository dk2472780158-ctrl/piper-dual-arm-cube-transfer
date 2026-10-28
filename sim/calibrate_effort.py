"""Fit the simulated torque channel to the effort the real arms reported.

The policy reads `observation.state`, which interleaves joint position and
torque. The MuJoCo model's gravity load is two to five times the torque the real
servos report, so the torque half of the state arrives far outside the range the
policy was trained on. This replays the recorded demonstration through the
physics, collects the model's actuator torque per frame, and fits one affine
correction per joint against the recorded channel.

Writes sim/effort_calibration.json.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "scene.xml")
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
SIDES = ("left", "right")
START = np.asarray([0.0093, 0.0049, -0.0024, -0.0582, 0.3185, 0.0973, 0.0003,
                    0.0631, 0.0048, -0.0027, -0.0766, 0.3512, 0.0202, 0.0])
np.set_printoptions(precision=3, suppress=True)

df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(REAL))])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)
states = np.stack(df["observation.state"].values)
recorded = states[:, 1::2]

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)


def drive(pose14):
    for side, base in (("left", 0), ("right", 7)):
        pose = pose14[base:base + 7]
        for i in range(1, 7):
            d.ctrl[m.actuator(f"{side}_act_{i}").id] = SIGN[i] * float(pose[i - 1])
        grip = float(np.clip(pose[6], 0.0, 0.07))
        d.ctrl[m.actuator(f"{side}_act_7").id] = grip / 2
        d.ctrl[m.actuator(f"{side}_act_8").id] = -grip / 2


mujoco.mj_resetData(m, d)
for i in range(1, 7):
    for side, base in (("left", 0), ("right", 7)):
        value = SIGN[i] * START[base + i - 1]
        d.qpos[int(np.ravel(m.joint(f"{side}_joint{i}").qposadr)[0])] = value
mujoco.mj_forward(m, d)

model_torque = []
for f in range(len(actions)):
    drive(actions[f])
    for _ in range(17):
        mujoco.mj_step(m, d)
    row = []
    for side in SIDES:
        for i in range(1, 7):
            row.append(SIGN[i] * float(d.qfrc_actuator[
                m.joint(f"{side}_joint{i}").dofadr[0]]))
        row.append(float(d.qfrc_actuator[m.joint(f"{side}_joint7").dofadr[0]]))
    model_torque.append(row)
model_torque = np.asarray(model_torque)

names = [f"{side}_{motor}" for side in SIDES
         for motor in ("j1", "j2", "j3", "j4", "j5", "j6", "grip")]
calibration = {}
print("motor      real mean   real std     sim mean    sim std     gain   offset"
      "     rms")
for k, name in enumerate(names):
    x = model_torque[:, k]
    y = recorded[:, k]
    design = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - (a * x + b)
    print(f"{name:10s} {y.mean():9.3f} {y.std():9.3f} {x.mean():11.3f} "
          f"{x.std():9.3f} {a:8.3f} {b:8.3f} {residual.std():7.3f}")
    calibration[name] = {"gain": float(a), "offset": float(b)}

with open(os.path.join(HERE, "effort_calibration.json"), "w", encoding="utf-8") as f:
    json.dump(calibration, f, indent=2)
print("wrote", os.path.join(HERE, "effort_calibration.json"))
