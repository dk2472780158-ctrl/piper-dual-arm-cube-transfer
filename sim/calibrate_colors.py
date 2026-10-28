"""Match the scene's material colours to the recorded frame.

MuJoCo multiplies the albedo by the lighting, so a material renders about 1.3x
its albedo under these two lights. Picking albedos that look right by eye
therefore leaves every surface over-bright and clips the bright ones.

Colour thresholds are not used to find the surfaces: a threshold that finds the
mat in the recording also drops the brightest part of the same mat in a render,
which is what makes the measured colour drift. Instead each surface is a fixed
disc at the recorded position -- the two cameras agree to a few pixels -- and
the median colour inside that disc is compared between the recording and a
render of the same arm pose.

Prints the albedo that would reproduce the recorded rgb.
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
np.set_printoptions(precision=3, suppress=True)

# Measured on the recorded middle frame 0: (centre x, centre y, radius).
DISCS = {
    "mat_yellow": (243.0, 300.6, 50),
    "mat_purple": (500.7, 304.2, 50),
    "cube_pink": (500.9, 279.9, 11),
}
CUBE_DISC = DISCS["cube_pink"]


def disc_mask(cx, cy, r):
    ys, xs = np.mgrid[0:480, 0:640]
    return (xs - cx) ** 2 + (ys - cy) ** 2 < r * r


def sample(image):
    rgb = np.asarray(image)[:, :, :3].astype(float) / 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    out = {}
    for name, (cx, cy, r) in DISCS.items():
        if name == "cube_pink":
            # The cube is too small for a fixed disc: it moves in the image when
            # anything about its height changes, and a disc then averages the
            # lit top face with the shaded sides in a different proportion. A
            # colour mask covers the same faces of the cube in both images.
            mask = ((red > 0.45) & (red - green > 0.12) & (red - blue > 0.05)
                    & disc_mask(cx, cy, 40))
        else:
            mask = disc_mask(cx, cy, r) & ~disc_mask(*CUBE_DISC)
        out[name] = np.median(rgb[mask], axis=0)
    out["tabletop"] = np.median(rgb[250:340, 340:400].reshape(-1, 3), axis=0)
    return out


real_image = imageio.imread("/tmp/real_middle.png")
real = sample(real_image)

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
# Read the albedos from the scene rather than keeping a copy here: the copy
# drifts from the file as soon as one surface is calibrated twice.
CURRENT = {name: model.mat_rgba[model.material(name).id][:3].copy()
           for name in DISCS}
CURRENT["tabletop"] = model.mat_rgba[model.material("tabletop").id][:3].copy()
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(
    f"{DATASET}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
state = np.stack(df["observation.state"].values)[0][0::2]
for side, base in (("left", 0), ("right", 7)):
    for i in range(1, 7):
        data.qpos[int(np.ravel(model.joint(f"{side}_joint{i}").qposadr)[0])] = (
            SIGN[i] * state[base + i - 1])
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, 480, 640)
renderer.update_scene(data, camera="middle")
sim_image = renderer.render()
sim = sample(sim_image)


def sharpness(image):
    """Mean absolute gradient: the recording is compressed and slightly soft,
    so a sharp render is a visible difference in itself."""
    grey = np.asarray(image)[:, :, :3].astype(float).mean(axis=2) / 255.0
    return float(np.abs(np.diff(grey, axis=0)).mean() + np.abs(np.diff(grey, axis=1)).mean())


print(f"sharpness  recorded {sharpness(real_image):.4f}  rendered {sharpness(sim_image):.4f}")

print("surface      recorded rgb          rendered rgb          ratio"
      "      new albedo")
for name, current in CURRENT.items():
    ratio = np.clip(real[name] / np.maximum(sim[name], 0.02), 0.2, 1.6)
    new = np.clip(current * ratio, 0.02, 1.0)
    print(f"{name:11s} {np.round(real[name], 3)}  {np.round(sim[name], 3)}  "
          f"{np.round(ratio, 2)}  rgba=\"{new[0]:.3f} {new[1]:.3f} {new[2]:.3f} 1\"")
