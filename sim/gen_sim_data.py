"""Record training episodes in the simulated cell.

The recorded demonstration is a position trajectory, so it stays a usable
expert under small perturbations: let the arm start a little off, or let the
commanded action wander, and the stream pulls the arm back onto the recorded
path. That gives behaviour cloning something the closed loop needs and a pure
replay cannot supply -- what to do when the state is not exactly nominal.

Each episode randomises the things the real cell does not hold fixed either:
surface colours, the two lights, where the cube was put down, its mass and
friction, and the two arms' start pose. The scene is otherwise the fitted one,
so the frames come out of the same pipeline the policy will see at run time
(render, then the fitted camera response).

Actions are recorded clean and commanded noisy, so a state that drifted is
labelled with the action that would have prevented the drift.

Every take runs twice: once with the cameras off to see whether it lands, and
only if it does, once with them on to record it. Physics does not depend on
rendering, so the recorded take is the same take.

    MUJOCO_GL=egl python gen_sim_data.py --episodes 60 --out cube_sim_v1
"""
import argparse
import copy
import glob
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mujoco

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from directions import DIRECTION, MAT_RADIUS, MAT_TOP, mirror, spawn, stream
from piper_dual_sim import PiperDualSim, JOINT_SIGN, MOTORS, SIDES

ROOT = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust")
TASK = "Place the cube to the other side."
CAMS = ("left", "middle", "right")
HEIGHT, WIDTH = 480, 640

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=60)
ap.add_argument("--out", default="cube_sim_v1")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--action-noise", type=float, default=0.010,
                help="rad per step, slow wander added to the working arm")
ap.add_argument("--start-jitter", type=float, default=0.015,
                help="rad, spread of the arms' start pose")
ap.add_argument("--cube-jitter", type=float, default=0.003,
                help="m, spread of the cube's place on the mat")
ap.add_argument("--retries", type=int, default=3,
                help="a take that misses is re-rolled this many times")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
df = pd.concat([pd.read_parquet(f)
                for f in sorted(glob.glob(f"{ROOT}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values).astype(np.float32)
HOME = actions[0].copy()
FRAMES = args.max_frames or len(actions)
STREAM = {name: stream(actions, name) for name in DIRECTION}

sim = PiperDualSim(render=True)
sim.connect()
model, data = sim._model, sim._data
cube_id = model.body("cube").id
cube_geom = model.geom("cube_geom").id
free = int(np.ravel(model.joint("cube_free").qposadr)[0])
materials = {name: model.material(name).id
             for name in ("tabletop", "mat_yellow", "mat_purple", "cube_pink",
                          "piper")}
base_rgba = {name: model.mat_rgba[index].copy()
             for name, index in materials.items()}
base_light = (model.light_diffuse.copy(), model.light_dir.copy(),
              model.light_ambient.copy())


def joint(side, index):
    return int(np.ravel(model.joint(f"{side}_joint{index}").qposadr)[0])


def reset(direction):
    mujoco.mj_resetData(model, data)
    home = HOME if direction == "r2l" else mirror(HOME)
    for side, base in (("left", 0), ("right", 7)):
        for i in range(1, 7):
            value = JOINT_SIGN[i] * float(home[base + i - 1])
            value += float(rng.normal(0.0, args.start_jitter))
            data.qpos[joint(side, i)] = value
            data.ctrl[sim._actuator_ids[(side, i)]] = value
        grip = float(np.clip(home[base + 6], 0.0, 0.07))
        data.qpos[joint(side, 7)] = grip / 2
        data.qpos[joint(side, 8)] = -grip / 2
        data.ctrl[sim._actuator_ids[(side, 7)]] = grip / 2
        data.ctrl[sim._actuator_ids[(side, 8)]] = -grip / 2
    x, y, z, qw, _, _, qz = spawn(direction)
    half = math.copysign(math.acos(max(-1.0, min(1.0, qw))), qz)
    half += float(rng.normal(0.0, 0.06))
    data.qpos[free:free + 7] = (x + float(rng.normal(0.0, args.cube_jitter)),
                                y + float(rng.normal(0.0, args.cube_jitter)),
                                z, math.cos(half), 0.0, 0.0, math.sin(half))
    mujoco.mj_forward(model, data)


def randomize_look():
    """Surfaces and lights, per episode: what the policy must not depend on."""
    for name, index in materials.items():
        rgba = base_rgba[name].copy()
        rgba[:3] = np.clip(rgba[:3] * rng.uniform(0.80, 1.20, 3), 0, 1)
        model.mat_rgba[index] = rgba
    diffuse, direction, ambient = (value.copy() for value in base_light)
    diffuse *= rng.uniform(0.75, 1.25, diffuse.shape)
    ambient *= rng.uniform(0.70, 1.30, ambient.shape)
    model.light_diffuse[:] = np.clip(diffuse, 0, 1)
    model.light_ambient[:] = np.clip(ambient, 0, 1)
    model.light_dir[:] = direction + rng.normal(0.0, 0.05, direction.shape)
    model.body_mass[cube_id] = float(rng.uniform(0.027, 0.033))
    model.geom_friction[cube_geom] = np.array(
        [float(rng.uniform(1.2, 1.6)), 0.05, 0.001])


def state_vector(obs):
    return np.array([obs[f"{side}_{motor}.{part}"] for side in SIDES
                     for motor in MOTORS for part in ("pos", "effort")],
                    dtype=np.float32)


def action_dict(action14):
    return {f"{side}_{motor}.pos": float(action14[base + index])
            for side, base in (("left", 0), ("right", 7))
            for index, motor in enumerate(MOTORS)}


dataset = None


def run_take(direction, record):
    """Drive the demonstration once; label it with the clean action."""
    randomize_look()
    reset(direction)
    worker = DIRECTION[direction]["worker"]
    base = 0 if worker == "left" else 7
    stream_direction = STREAM[direction]
    noise = np.zeros(14, dtype=np.float64)
    for step in range(FRAMES):
        obs = sim.get_observation()
        clean = stream_direction[min(step, len(stream_direction) - 1)]
        if record:
            frame = {"observation.state": state_vector(obs),
                     "action": clean.astype(np.float32), "task": TASK}
            for name in CAMS:
                frame[f"observation.images.{name}"] = np.asarray(obs[name])
            dataset.add_frame(frame)
        noise = 0.9 * noise + rng.normal(0.0, args.action_noise, 14)
        commanded = clean.copy()
        commanded[base:base + 6] += noise[base:base + 6].astype(np.float32)
        sim.send_action(action_dict(commanded))
    sim._advance()                     # settle the last commanded target
    end = data.xpos[cube_id].copy()
    offset = float(np.linalg.norm(end[:2] - np.asarray(
        DIRECTION[direction]["target"])))
    held = float(data.qpos[joint(worker, 7)] - data.qpos[joint(worker, 8)])
    return (bool(offset < MAT_RADIUS and end[2] < MAT_TOP + 0.033 and held > 0.02),
            offset)


features = {
    "observation.state": {**sim.motor_features["observation.state"],
                          "dtype": "float32", "info": None},
    "action": {**sim.motor_features["action"], "dtype": "float32", "info": None},
}
for name in CAMS:
    features[f"observation.images.{name}"] = {
        "dtype": "video", "shape": (HEIGHT, WIDTH, 3),
        "names": ["height", "width", "channels"], "info": None}

root = Path("/home/databall/.cache/huggingface/lerobot/local") / args.out
dataset = LeRobotDataset.create(
    f"local/{args.out}", 30, features=features, root=root,
    robot_type="piper_dual", use_videos=True, batch_encoding_size=1,
    vcodec="h264")

started = time.monotonic()
kept = attempts = 0
while kept < args.episodes:
    direction = "r2l" if kept % 2 == 0 else "l2r"
    landed = False
    for _ in range(args.retries):
        attempts += 1
        # the same draw has to run twice: once blind, once recording
        snapshot = copy.deepcopy(rng.bit_generator.state)
        sim._render_enabled = False
        landed, offset = run_take(direction, record=False)
        if not landed:
            continue
        rng.bit_generator.state = snapshot
        sim._render_enabled = True
        landed, offset = run_take(direction, record=True)
        break
    if not landed:
        print(f"  dropped a {direction} take ({100 * offset:.1f} cm from the "
              f"mat centre)", flush=True)
        continue
    dataset.save_episode()
    kept += 1
    print(f"  episode {kept}/{args.episodes} {direction}  "
          f"{100 * offset:4.1f} cm from centre  "
          f"{(time.monotonic() - started) / kept:5.1f} s/episode  "
          f"{attempts} takes", flush=True)

dataset.finalize()
sim.disconnect()
print(f"wrote {root}  episodes {kept}  takes {attempts}")
