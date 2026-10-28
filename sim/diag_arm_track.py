"""Does the simulated arm follow the recorded one as the arm moves?

At the start of an episode every joint is near zero, so a wrong link length, a
wrong joint sign or a wrong base pose all hide. They stop hiding once the arm
is reaching. This drives the model from the recorded joint states and compares
the dark arm silhouette (centre, extent, area) in the middle view, frame by
frame, so a drift shows up as a trend rather than as one bad picture.

Usage:  MUJOCO_GL=egl python diag_arm_track.py
"""
import glob

import numpy as np
import pandas as pd
import imageio.v2 as imageio
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
DATASET = ("/home/databall/.cache/huggingface/lerobot/local/"
           "cube_r2l_dataset_v2_robust")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
DARK = 0.32
np.set_printoptions(precision=3, suppress=True)

df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(
    f"{DATASET}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
states = np.stack(df["observation.state"].values)[:, 0::2]

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 480, 640)
reader = imageio.get_reader(glob.glob(
    f"{DATASET}/videos/observation.images.middle/*/*.mp4")[0])


def dark(image):
    rgb = np.asarray(image)[:, :, :3].astype(float) / 255.0
    return rgb.max(axis=2) < DARK


def describe(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    return np.array([xs.mean(), ys.mean(), xs.max() - xs.min(),
                     ys.max() - ys.min(), float(len(xs))])


print("frame | real cx cy w h area | sim cx cy w h area | delta")
for frame in range(0, 260, 20):
    for side, base in (("left", 0), ("right", 7)):
        for i in range(1, 7):
            data.qpos[int(np.ravel(model.joint(f"{side}_joint{i}").qposadr)[0])] = (
                SIGN[i] * states[frame][base + i - 1])
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="middle")
    sim = renderer.render()
    real_d = describe(dark(reader.get_data(frame)))
    sim_d = describe(dark(sim))
    if real_d is None or sim_d is None:
        print(f"{frame:5d} | mask missing")
        continue
    print(f"{frame:5d} | {np.round(real_d[:4], 0)} {real_d[4]:7.0f} | "
          f"{np.round(sim_d[:4], 0)} {sim_d[4]:7.0f} | "
          f"dx {sim_d[0] - real_d[0]:+6.1f} dy {sim_d[1] - real_d[1]:+6.1f} "
          f"area x{sim_d[4] / real_d[4]:.2f}")
