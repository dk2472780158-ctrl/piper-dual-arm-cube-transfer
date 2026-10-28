"""Run the cube transfer over and over and score every attempt.

One episode is: reset the cell, drive one arm across the table, let go, and
check where the cube ended up. `--direction r2l` sends the cube from the purple
mat to the yellow one with the right arm, `--direction l2r` sends it back with
the left arm, and `--direction alternate` (the default) does one and then the
other, so the cell runs the same round trip the real one does.

There is a recording only for the right-to-left pass. The left-to-right pass is
that same recording reflected in y, which is exact: the mounts, the mats and the
cube are all placed as mirror images, and `diag_mirror.py` finds the one joint
sign map that puts the left fingertip on the mirror of the right one (0.0 mm,
0.0 deg on every frame it tried). Between two episodes the cube is set back on
the spot its arm expects, the way a person would with the mats as parking spots.

Two drivers are available. `demo` replays the recorded action column through
the physics, which is the replica test. `policy` closes the loop around the ACT
checkpoint, feeding the rendered cameras back in and using whatever the policy
asks for.

Every observation comes from PiperDualSim, the same object the recording
pipeline uses, so the policy here sees exactly what it was trained on: SDK joint
signs, calibrated torques, and the fitted camera response. Reading qpos and
qfrc_actuator straight out of MuJoCo instead skips all three.

    python loop.py                          # 5 episodes, alternating
    python loop.py --episodes 20 --direction r2l
    python loop.py --episodes 2 --direction alternate --video /tmp/ep.mp4
    python loop.py --episodes 3 --driver policy

Success is the cube resting inside the target mat and the gripper open.
"""
import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mujoco

from directions import DIRECTION, MAT_RADIUS, MAT_TOP, mirror, spawn, stream
from piper_dual_sim import PiperDualSim, JOINT_SIGN, MOTORS, SIDES

# Where the demonstration comes from, and the checkpoint the real cell trained.
DEMO_ROOT = ("/home/databall/.cache/huggingface/lerobot/local/"
             "cube_r2l_dataset_v2_robust")
REAL_CKPT = ("/home/databall/VLA/lerobot_piper/outputs/train/"
             "cube_r2l_new_env_ft_from_80ep20k/checkpoints/020000/"
             "pretrained_model")
# The checkpoint that passes here: ACT fine-tuned on episodes recorded in the
# simulation. The real-cell one drops the cube -- it was trained on photographs.
SIM_CKPT = ("/home/databall/VLA/lerobot_piper/outputs/train/"
            "cube_sim_act_v1_ft/checkpoints/012000/pretrained_model")
SIM_ROOT = "/home/databall/.cache/huggingface/lerobot/local/cube_sim_v1"
TASK = "Place the cube to the other side."
CAMS = ("left", "middle", "right")
np.set_printoptions(precision=4, suppress=True)

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=5)
ap.add_argument("--direction", choices=("r2l", "l2r", "alternate"),
                default="alternate")
ap.add_argument("--driver", choices=("demo", "policy"), default="demo")
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--video", default="", help="record the whole run to this mp4")
ap.add_argument("--max-frames", type=int, default=0,
                help="stop an episode after this many frames; 0 uses the recording")
# ACT was trained to emit a chunk and consume it for a while; 1 with
# re-planning every frame plus the temporal ensembler is the other common
# choice. Which one the checkpoint expects is a deployment question.
ap.add_argument("--n-action-steps", type=int, default=1)
ap.add_argument("--ensemble", type=float, default=0.01,
                help="temporal ensembling coefficient; <=0 turns it off")
ap.add_argument("--scene", default="scene_real.xml",
                help="scene file to run; scene_real.xml is the textured, "
                     "shadow-casting variant, scene.xml the flat original")
ap.add_argument("--ckpt", default=SIM_CKPT, help="ACT checkpoint to drive with")
ap.add_argument("--ds-root", default=SIM_ROOT,
                help="dataset whose metadata describes the policy's features")
args = ap.parse_args()

# The checkpoint trained on the recorded cell knows the right-to-left pass
# only. A checkpoint trained on the simulated set saw both directions, so the
# restriction follows the checkpoint, not the driver.
if args.driver == "policy" and args.direction != "r2l" and args.ckpt == REAL_CKPT:
    print("this checkpoint was trained on the right-to-left pass; "
          "running r2l only.")
    args.direction = "r2l"

df = pd.concat([pd.read_parquet(f) for f in
                sorted(glob.glob(f"{DEMO_ROOT}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)
HOME = actions[0].copy()
FRAMES = args.max_frames or len(actions)


STREAM = {name: stream(actions, name) for name in DIRECTION}

policy = pre = post = cfg = None
if args.driver == "policy":
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.control_utils import predict_action

    print(f"loading the ACT checkpoint from {args.ckpt} ...")
    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.pretrained_path = args.ckpt
    cfg.device = "cuda"
    cfg.n_action_steps = args.n_action_steps
    cfg.n_obs_steps = 1
    cfg.temporal_ensemble_coeff = (args.ensemble if args.ensemble > 0 else None)
    # ACT refuses n_obs_steps > 1; the queue length has to fit the chunk.
    cfg.n_action_steps = min(cfg.n_action_steps, cfg.chunk_size)
    policy = make_policy(cfg, ds_meta=LeRobotDatasetMetadata(
        f"local/{os.path.basename(args.ds_root.rstrip('/'))}",
        root=args.ds_root))
    pre, post = make_pre_post_processors(policy_cfg=cfg,
                                         pretrained_path=args.ckpt,
                                         dataset_stats=None)
    policy.eval()

sim = PiperDualSim(scene=Path(__file__).resolve().parent / args.scene,
                   render=args.driver == "policy")
sim.connect()
model, data = sim._model, sim._data
cube = model.body("cube").id
free = int(np.ravel(model.joint("cube_free").qposadr)[0])


def joint(side, index):
    return int(np.ravel(model.joint(f"{side}_joint{index}").qposadr)[0])


def reset(direction):
    mujoco.mj_resetData(model, data)
    home = HOME if direction == "r2l" else mirror(HOME)
    for side, base in (("left", 0), ("right", 7)):
        for i in range(1, 7):
            value = JOINT_SIGN[i] * float(home[base + i - 1])
            data.qpos[joint(side, i)] = value
            data.ctrl[sim._actuator_ids[(side, i)]] = value
        grip = float(np.clip(home[base + 6], 0.0, 0.07))
        data.qpos[joint(side, 7)] = grip / 2
        data.qpos[joint(side, 8)] = -grip / 2
        data.ctrl[sim._actuator_ids[(side, 7)]] = grip / 2
        data.ctrl[sim._actuator_ids[(side, 8)]] = -grip / 2
    data.qpos[free:free + 7] = spawn(direction)
    mujoco.mj_forward(model, data)


def state_vector(obs):
    return np.array([obs[f"{side}_{motor}.{part}"] for side in SIDES
                     for motor in MOTORS for part in ("pos", "effort")],
                    dtype=np.float32)


def action_dict(action14):
    return {f"{side}_{motor}.pos": float(action14[base + index])
            for side, base in (("left", 0), ("right", 7))
            for index, motor in enumerate(MOTORS)}


def policy_action(obs):
    observation_dict = {"observation.state": state_vector(obs)}
    for name in CAMS:
        observation_dict[f"observation.images.{name}"] = np.asarray(obs[name])
    return np.asarray(predict_action(
        observation=observation_dict, policy=policy,
        device=torch.device("cuda"), preprocessor=pre, postprocessor=post,
        use_amp=cfg.use_amp, task=TASK, robot_type="piper_dual")).reshape(-1)


def run_episode(direction, video=None):
    worker = DIRECTION[direction]["worker"]
    target = DIRECTION[direction]["target"]
    base = 0 if worker == "left" else 7
    start = data.xpos[cube].copy()
    writer, overview, renderer = video
    stream_direction = STREAM[direction]
    grip_cmd = float(stream_direction[0][base + 6])
    if args.driver == "policy":
        # A chunk-queueing policy carries its leftovers into the next episode:
        # without this the first 50 commands of episode N+1 come from the end
        # of episode N and the arm never leaves the previous mat.
        policy.reset()
        for processor in (pre, post):
            if hasattr(processor, "reset"):
                processor.reset()
    for step in range(FRAMES):
        loop_start = time.monotonic()
        obs = sim.get_observation()      # advances one control period
        if args.driver == "policy":
            action = policy_action(obs)
        else:
            action = stream_direction[min(step, len(stream_direction) - 1)]
        sim.send_action(action_dict(action))
        grip_cmd = float(action[base + 6])
        if writer is not None:
            renderer.update_scene(data, camera=overview)
            writer.append_data(renderer.render()[:, :, :3])
        if args.driver == "policy":
            time.sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - loop_start)))
    sim._advance()                       # settle the last commanded target
    end = data.xpos[cube].copy()
    offset = float(np.linalg.norm(end[:2] - np.asarray(target)))
    travel = float(np.linalg.norm(end[:2] - start[:2]))
    # the cube has to end inside the mat and be free of the fingers
    still_held = float(data.qpos[joint(worker, 7)] - data.qpos[joint(worker, 8)])
    ok = bool(offset < MAT_RADIUS and end[2] < MAT_TOP + 0.033 and still_held > 0.02)
    return ok, end, travel, offset, grip_cmd


order = []
for episode in range(args.episodes):
    if args.direction == "alternate":
        order.append("r2l" if episode % 2 == 0 else "l2r")
    else:
        order.append(args.direction)

print(f"driver {args.driver}, {args.episodes} episodes, {FRAMES} frames each, "
      f"order {'/'.join(order)}")
# --video records the whole run, so a two-direction loop comes out as one file.
writer = overview = renderer = None
if args.video:
    import imageio.v2 as imageio
    writer = imageio.get_writer(args.video, fps=int(args.fps))
    overview = mujoco.MjvCamera()
    overview.type = mujoco.mjtCamera.mjCAMERA_FREE
    overview.lookat[:] = (0.0, 0.0, 0.78)
    overview.distance = 1.9
    overview.azimuth = 145.0
    overview.elevation = -22.0
    renderer = mujoco.Renderer(model, 480, 640)
results = []
for episode, direction in enumerate(order):
    reset(direction)
    ok, end, travel, offset, grip_cmd = run_episode(
        direction, (writer, overview, renderer))
    results.append(ok)
    target = "yellow" if direction == "r2l" else "purple"
    print(f"  episode {episode + 1:2d} {direction}: {'PASS' if ok else 'FAIL'}  "
          f"cube {np.round(end, 4)}  travel {100 * travel:5.1f} cm  "
          f"{100 * offset:5.1f} cm from the {target} mat centre  "
          f"final grip cmd {grip_cmd:.3f}", flush=True)

if writer is not None:
    writer.close()
    print(f"the whole run is in {args.video}")
if renderer is not None:
    renderer.close()
sim.disconnect()
passed = sum(results)
print(f"\n{passed}/{len(results)} passed  ({100 * passed / len(results):.0f}%)")
for direction in ("r2l", "l2r"):
    hits = [ok for ok, d in zip(results, order) if d == direction]
    if hits:
        print(f"  {direction}: {sum(hits)}/{len(hits)}")
