"""How much of the remaining difference is the camera, not the cell?

The scene is now geometrically right: the mats, the cube and the gripper sit
where the recording puts them. What is left is the sensor -- the recorded
cameras are dimmer, darker at the frame edge and noisier than an EGL render.

This reads two recorded frames per camera together with the matching renders,
and fits one smooth per-channel gain field per camera:

    recorded ~= render * (a + b u + c v + d u^2 + e u v + f v^2) + g

u and v are pixel coordinates scaled to [-1, 1] about the frame centre. A gain
field expressed in the image plane is a property of the lens and the sensor, so
one fit covers every frame of a take; a scene-light model would not. Fitting on
two poses far apart in the episode keeps it from absorbing the contents of
either one.

Usage:  MUJOCO_GL=egl python fit_render.py [--write]
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
import imageio.v2 as imageio
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
DATASET = ("/home/databall/.cache/huggingface/lerobot/local/"
           "cube_r2l_dataset_v2_robust")
OUT = "/home/databall/VLA/cube_mujoco/sim/render_model.json"
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
# Fitted on early frames only. Later in the episode the two pictures differ in
# content -- the cube is in the gripper and the jaws are around it -- and a
# fit that includes those frames stops describing the camera and starts
# compensating for the cell. The frames below are all before the approach.
FRAMES = (0, 20, 40)
VALIDATE = 150
W, H = 640, 480

u = (np.arange(W) - (W - 1) / 2) / ((W - 1) / 2)
v = (np.arange(H) - (H - 1) / 2) / ((H - 1) / 2)
U, V = np.meshgrid(u, v)
# The multiplicative field varies over the frame; the additive term is a plain
# constant, so the two are not the same shape and the design is built per
# channel -- with a bare column of ones, not ones times the render, or the two
# would be collinear and the solve would split the gain arbitrarily.
BASIS = np.stack([np.ones_like(U), U, V, U * U, U * V, V * V], axis=-1)

df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(
    f"{DATASET}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
states = np.stack(df["observation.state"].values)[:, 0::2]

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, H, W)


def pose_arm(state14):
    for side, base in (("left", 0), ("right", 7)):
        for i in range(1, 7):
            data.qpos[int(np.ravel(model.joint(f"{side}_joint{i}").qposadr)[0])] = (
                SIGN[i] * state14[base + i - 1])
    mujoco.mj_forward(model, data)


def pairs(camera):
    """(render, recorded) pixel stacks over the chosen frames."""
    reader = imageio.get_reader(glob.glob(
        f"{DATASET}/videos/observation.images.{camera}/*/*.mp4")[0])
    renders, records = [], []
    for frame in FRAMES:
        pose_arm(states[frame])
        renderer.update_scene(data, camera=camera)
        renders.append(renderer.render()[:, :, :3].astype(float))
        records.append(np.asarray(reader.get_data(frame))[:, :, :3].astype(float))
    return renders, records


out = {}
for camera in ("left", "middle", "right"):
    renders, records = pairs(camera)
    raw = [np.abs(r - s).mean() for r, s in zip(records, renders)]
    before = float(np.mean(raw))
    # Flat design so the six coefficients solve independently for each channel.
    coef = np.zeros((3, 6))
    for c in range(3):
        rows = np.concatenate([(BASIS * s[..., c][..., None]).reshape(-1, 6)
                               for s in renders])
        values = np.concatenate([r[..., c].reshape(-1) for r in records])
        coef[c], *_ = np.linalg.lstsq(rows, values, rcond=None)
    # The arm and the cube do not match the recording exactly at every fitted
    # frame, so the first solve is pulled by the pixels that differ in content
    # rather than in response. Two reweighting passes drop the worst residuals
    # -- a fifth, then a third of them -- and keep the fit on the parts of the
    # picture the two agree on.
    for keep in (0.8, 0.66):
        masks = []
        for real, sim in zip(records, renders):
            predicted = np.stack([
                (BASIS * sim[..., c][..., None]) @ coef[c] for c in range(3)],
                axis=-1)
            residual = np.abs(real - predicted).mean(axis=2).reshape(-1)
            masks.append(residual <= np.quantile(residual, keep))
        for c in range(3):
            rows, values = [], []
            for real, sim, mask in zip(records, renders, masks):
                block = (BASIS * sim[..., c][..., None]).reshape(-1, 6)
                rows.append(block[mask])
                values.append(real[..., c].reshape(-1)[mask])
            coef[c], *_ = np.linalg.lstsq(np.concatenate(rows),
                                          np.concatenate(values), rcond=None)
    after = []
    for real, sim in zip(records, renders):
        fixed = np.stack([
            (BASIS * sim[..., c][..., None]) @ coef[c] for c in range(3)],
            axis=-1)
        after.append(np.abs(real - np.clip(fixed, 0, 255)).mean())
    per_frame = " ".join(
        f"f{f} {b:.1f}->{a:.1f}" for f, b, a in zip(FRAMES, raw, after))
    print(f"{camera:7s} mean |diff| {before:6.2f} -> {np.mean(after):6.2f}   {per_frame}")
    for c, name in enumerate("rgb"):
        print(f"   {name}: gain {coef[c, 0]:.3f} x {coef[c, 1]:+.3f} y "
              f"{coef[c, 2]:+.3f} u2 {coef[c, 3]:+.3f} uv {coef[c, 4]:+.3f} "
              f"v2 {coef[c, 5]:+.3f}")
    out[camera] = coef.tolist()

if "--write" in sys.argv:
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=2)
    print("wrote", OUT)
