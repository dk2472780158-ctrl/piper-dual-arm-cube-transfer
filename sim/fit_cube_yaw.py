"""Where exactly was the cube placed, and at what angle?

The cell description spawns the cube at the centre of the purple mat and
axis-aligned. In the recording it was set down by hand: its centre is about a
centimetre off the mat centre and it is turned. The middle camera sees it
cleanly enough to read both -- the position from the centre of its colour mask,
the angle from the overlap of the mask -- and the policy sees both.

Usage:  MUJOCO_GL=egl python fit_cube_yaw.py
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
np.set_printoptions(precision=4, suppress=True)


def cube_mask(image):
    rgb = np.asarray(image)[:, :, :3].astype(float) / 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return ((red > 0.45) & (red - green > 0.12) & (red - blue > 0.05)
            & (green < 0.62))


def iou(a, b):
    union = float((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


reader = imageio.get_reader(glob.glob(
    f"{DATASET}/videos/observation.images.middle/*/*.mp4")[0])
real = cube_mask(np.asarray(reader.get_data(0)))
ys, xs = np.nonzero(real)
REAL_CENTRE = np.array([xs.mean(), ys.mean()])
print(f"recorded cube centre {np.round(REAL_CENTRE, 1)} area {real.sum()}")

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(
    f"{DATASET}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
state = np.stack(df["observation.state"].values)[0][0::2]
for side, base in (("left", 0), ("right", 7)):
    for i in range(1, 7):
        data.qpos[int(np.ravel(model.joint(f"{side}_joint{i}").qposadr)[0])] = (
            SIGN[i] * state[base + i - 1])

free = model.joint("cube_free").id
qadr = int(np.ravel(model.jnt_qposadr[free])[0])
spawn = data.qpos[qadr:qadr + 3].copy()
renderer = mujoco.Renderer(model, 480, 640)


def measure(offset, yaw):
    data.qpos[qadr:qadr + 3] = spawn + [offset[0], offset[1], 0.0]
    half = np.radians(yaw) / 2
    data.qpos[qadr + 3:qadr + 7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="middle")
    mask = cube_mask(renderer.render())
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        return np.array([1e3, 1e3]), 0.0
    return np.array([xs.mean(), ys.mean()]), iou(mask, real)


def centre_error(offset):
    centre, _ = measure(offset, 0.0)
    return float(np.linalg.norm(centre - REAL_CENTRE))


best_offset = np.zeros(2)
for span, step in ((0.03, 0.01), (0.012, 0.003)):
    grid = np.arange(-span, span + 1e-9, step)
    scores = [(centre_error(best_offset + np.array([dx, dy])), np.array([dx, dy]))
              for dx in grid for dy in grid]
    error, delta = min(scores, key=lambda item: item[0])
    best_offset = best_offset + delta
    print(f"offset {np.round(best_offset, 4)}  centre error {error:.1f} px")
print(f"offset from the mat centre {np.round(best_offset, 4)} m")

print("\nyaw_deg  IoU")
best_yaw, best_iou = 0.0, 0.0
for yaw in range(0, 90, 5):
    _, score = measure(best_offset, yaw)
    print(f"{yaw:6d}  {score:.4f}")
    if score > best_iou:
        best_yaw, best_iou = float(yaw), score
print(f"best yaw {best_yaw:.0f} deg, IoU {best_iou:.4f}")
centre, _ = measure(best_offset, best_yaw)
print(f"cube centre at the fit {np.round(centre, 1)} (recorded {np.round(REAL_CENTRE, 1)})")
