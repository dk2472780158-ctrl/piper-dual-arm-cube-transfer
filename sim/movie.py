"""One video of the whole cell: the arms, the table, and what every camera sees.

The wrist views alone say nothing about whether the arms are where they should
be. This renders the cell from a free camera standing off the table, with the
three recorded camera views alongside it, so the pose of both arms, the mats and
the cube are all visible at once.

    MUJOCO_GL=egl python movie.py                 # the recorded demonstration
    MUJOCO_GL=egl python movie.py --run <dir>     # a policy run, replaying its actions

The actions are replayed through the physics, so the cube moves (or fails to)
exactly as it does in a policy evaluation.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import imageio.v2 as imageio
import mujoco

import render_sensor

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
DATASET = ("/home/databall/.cache/huggingface/lerobot/local/"
           "cube_r2l_dataset_v2_robust")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
CAMERAS = ("left", "middle", "right")
np.set_printoptions(precision=4, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="", help="policy run dir; default is the demo")
parser.add_argument("--stride", type=int, default=2, help="keep one frame in N")
parser.add_argument("--out", default="/tmp/cell.mp4")
parser.add_argument("--azimuth", type=float, default=145.0)
parser.add_argument("--elevation", type=float, default=-22.0)
parser.add_argument("--distance", type=float, default=1.9)
parser.add_argument("--no-panels", action="store_true",
                    help="full cell only, no camera insets")
args = parser.parse_args()

if args.run:
    files = sorted(glob.glob(f"{args.run}/data/chunk-*/file-*.parquet"))
    frame_source = os.path.basename(args.run.rstrip("/"))
else:
    files = sorted(glob.glob(f"{DATASET}/data/chunk-*/file-*.parquet"))
    frame_source = "recorded demonstration"
df = pd.concat([pd.read_parquet(f) for f in files]).reset_index(drop=True)
if not args.run:
    df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)
print(f"{frame_source}: {len(actions)} frames")

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
cube = model.body("cube").id
mujoco.mj_forward(model, data)
start = data.xpos[cube].copy()

overview = mujoco.MjvCamera()
overview.type = mujoco.mjtCamera.mjCAMERA_FREE
overview.lookat[:] = (0.0, 0.0, 0.78)
overview.distance = args.distance
overview.azimuth = args.azimuth
overview.elevation = args.elevation

renderer = mujoco.Renderer(model, 480, 640)
sensor = render_sensor.load()
writer = imageio.get_writer(args.out, fps=max(1, 30 // args.stride))


def drive(pose14):
    for side, base in (("left", 0), ("right", 7)):
        for i in range(1, 7):
            data.ctrl[model.actuator(f"{side}_act_{i}").id] = (
                SIGN[i] * float(pose14[base + i - 1]))
        grip = float(np.clip(pose14[base + 6], 0.0, 0.07))
        data.ctrl[model.actuator(f"{side}_act_7").id] = grip / 2.0
        data.ctrl[model.actuator(f"{side}_act_8").id] = -grip / 2.0


written = 0
for f in range(len(actions)):
    drive(actions[f])
    for _ in range(17):
        mujoco.mj_step(model, data)
    if f % args.stride:
        continue
    renderer.update_scene(data, camera=overview)
    panels = [renderer.render()[:, :, :3]]
    if not args.no_panels:
        for name in CAMERAS:
            renderer.update_scene(data, camera=name)
            panels.append(render_sensor.apply(sensor, name, renderer.render())[:, :, :3])
    if len(panels) == 1:
        writer.append_data(panels[0])
    else:
        top = np.concatenate(panels[:2], axis=1)
        bottom = np.concatenate(panels[2:], axis=1)
        writer.append_data(np.concatenate([top, bottom], axis=0))
    written += 1

writer.close()
end = data.xpos[cube].copy()
print(f"wrote {args.out}  {written} frames  ({args.out.rsplit('.', 1)[-1]})")
print("cube start", np.round(start, 4), " end", np.round(end, 4))
print("travel %.1f cm, lift %.1f cm, reached yellow mat (y > 0.05): %s"
      % (100 * np.linalg.norm((end - start)[:2]), 100 * (end[2] - start[2]),
         bool(end[1] > 0.05)))
