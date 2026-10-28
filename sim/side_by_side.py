"""Recorded frame next to a render of the same arm pose, per camera.

The numbers in calibrate_colors / fit_wrist say whether the silhouettes agree;
this is the picture to check that nothing else is different for the wrong
reason.

Usage:  MUJOCO_GL=egl python side_by_side.py [frame]
"""
import glob
import sys

import numpy as np
import pandas as pd
import imageio.v2 as imageio
import mujoco

import render_sensor

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
DATASET = ("/home/databall/.cache/huggingface/lerobot/local/"
           "cube_r2l_dataset_v2_robust")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 0

df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(
    f"{DATASET}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
state = np.stack(df["observation.state"].values)[FRAME][0::2]

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
for side, base in (("left", 0), ("right", 7)):
    for i in range(1, 7):
        data.qpos[int(np.ravel(model.joint(f"{side}_joint{i}").qposadr)[0])] = (
            SIGN[i] * state[base + i - 1])
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, 480, 640)
sensor = render_sensor.load()
rows = []
for camera in ("left", "middle", "right"):
    reader = imageio.get_reader(glob.glob(
        f"{DATASET}/videos/observation.images.{camera}/*/*.mp4")[0])
    real = np.asarray(reader.get_data(FRAME))[:, :, :3]
    renderer.update_scene(data, camera=camera)
    rows.append(np.concatenate(
        [real, render_sensor.apply(sensor, camera, renderer.render())], axis=1))
imageio.imwrite(f"/tmp/side_by_side_{FRAME}.png",
                np.concatenate(rows, axis=0))
print("wrote", f"/tmp/side_by_side_{FRAME}.png", "rows: left / middle / right")
