"""Ground-truth FK from the Piper SDK, applied to the recorded episode.

The SDK ships a DH model for the arm, so an end-effector pose can be computed
from a joint vector without trusting our URDF conversion. `CalFK` returns one
6-vector [x, y, z, roll, pitch, yaw] per link, in mm and degrees; the last entry
is the flange (link 6).
"""
import glob

import numpy as np
import pandas as pd
from piper_sdk import C_PiperForwardKinematics

REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")

np.set_printoptions(precision=4, suppress=True)

path = sorted(glob.glob(REAL))[0]
df = pd.read_parquet(path)
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)
states = np.stack(df["observation.state"].values)
fk = C_PiperForwardKinematics()

print("frames", len(df))


def flange(row6):
    return np.asarray(fk.CalFK([float(v) for v in row6])[-1])


for f in range(0, len(actions), 30):
    a = flange(actions[f][7:13])
    s = flange(states[f][14:20])
    print(f"f{f:4d} act {a.round(1)} grip {actions[f][13]:.3f} | state {s.round(1)}")

print()
for f in (0, 60, 120, 180, 185, 210, 300, 376, 437):
    a = flange(actions[f][7:13])
    print(f"f{f:4d} flange xyz {a[:3].round(2)} mm  rpy {a[3:].round(1)}  grip {actions[f][13]:.3f}")
