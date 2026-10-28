"""Compare simulated actuator effort with the recorded effort channel.

observation.state interleaves pos and effort per motor. The policy reads all
28 values, so if the sim's effort is far from the recorded range the policy is
being fed out-of-distribution input.
"""
import glob

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
np.set_printoptions(precision=3, suppress=True)

df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
states = np.stack(df["observation.state"].values)
real_eff = states[:, 1::2]

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
for side, base in (("left", 0), ("right", 7)):
    for i in range(1, 7):
        v = SIGN[i] * float(states[0][2 * (base + i - 1)])
        d.qpos[int(np.ravel(m.joint(f"{side}_joint{i}").qposadr)[0])] = v
        d.ctrl[m.actuator(f"{side}_act_{i}").id] = v
    g = float(states[0][2 * (base + 6)])
    d.qpos[int(np.ravel(m.joint(f"{side}_joint7").qposadr)[0])] = g / 2
    d.qpos[int(np.ravel(m.joint(f"{side}_joint8").qposadr)[0])] = -g / 2
    d.ctrl[m.actuator(f"{side}_act_7").id] = g / 2
    d.ctrl[m.actuator(f"{side}_act_8").id] = -g / 2
mujoco.mj_forward(m, d)
for _ in range(60):
    mujoco.mj_step(m, d)

sim_eff = []
for side in ("left", "right"):
    for i in range(1, 7):
        sim_eff.append(float(d.qfrc_actuator[m.joint(f"{side}_joint{i}").dofadr[0]]))
    sim_eff.append(float(d.qfrc_actuator[m.joint(f"{side}_joint7").dofadr[0]]))
sim_eff = np.asarray(sim_eff)

print("motor                real mean     real std   real min   real max    sim")
names = [f"{s}_{k}" for s in ("left", "right")
         for k in ("j1", "j2", "j3", "j4", "j5", "j6", "grip")]
for i, name in enumerate(names):
    print(f"{name:10s} {real_eff[:, i].mean():11.3f} {real_eff[:, i].std():11.3f} "
          f"{real_eff[:, i].min():10.3f} {real_eff[:, i].max():10.3f} "
          f"{sim_eff[i]:8.3f}")
print()
print("frame-0 real effort ", np.round(real_eff[0], 3))
print("sim effort after 2 s", np.round(sim_eff, 3))
