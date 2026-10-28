"""What does the policy ask for, and what did the demonstration ask for?

Runs one direction open loop -- the recorded stream still drives the arm -- and
prints both action columns side by side. Use it when a direction fails in the
closed loop: if the predictions already miss here, the policy never learned the
pass; if they match and the arm still goes wrong, the closed loop is losing the
state somewhere.

    MUJOCO_GL=egl python diag_policy_trace.py --direction l2r
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import mujoco

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.control_utils import predict_action

from directions import DIRECTION, mirror, spawn, stream
from piper_dual_sim import PiperDualSim, JOINT_SIGN, MOTORS, SIDES

ROOT = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust")
TASK = "Place the cube to the other side."
CAMS = ("left", "middle", "right")
np.set_printoptions(precision=3, suppress=True)

ap = argparse.ArgumentParser()
ap.add_argument("--direction", choices=("r2l", "l2r"), default="l2r")
ap.add_argument("--frames", type=int, default=438)
ap.add_argument("--step", type=int, default=20)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--ds-root", required=True)
ap.add_argument("--n-action-steps", type=int, default=50)
ap.add_argument("--closed-loop", action="store_true",
                help="drive with the prediction instead of the recording")
args = ap.parse_args()

df = pd.concat([pd.read_parquet(f)
                for f in sorted(glob.glob(f"{ROOT}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)
HOME = actions[0].copy()
stream_direction = stream(actions, args.direction)
base = 0 if DIRECTION[args.direction]["worker"] == "left" else 7

cfg = PreTrainedConfig.from_pretrained(args.ckpt)
cfg.pretrained_path = args.ckpt
cfg.device = "cuda"
cfg.n_action_steps = args.n_action_steps
cfg.n_obs_steps = 1
cfg.temporal_ensemble_coeff = None
policy = make_policy(cfg, ds_meta=LeRobotDatasetMetadata(
    f"local/{os.path.basename(args.ds_root.rstrip('/'))}", root=args.ds_root))
pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=args.ckpt,
                                     dataset_stats=None)
policy.eval()

sim = PiperDualSim(render=True)
sim.connect()
model, data = sim._model, sim._data
free = int(np.ravel(model.joint("cube_free").qposadr)[0])


def joint(side, index):
    return int(np.ravel(model.joint(f"{side}_joint{index}").qposadr)[0])


mujoco.mj_resetData(model, data)
home = HOME if args.direction == "r2l" else mirror(HOME)
for side, offset in (("left", 0), ("right", 7)):
    for i in range(1, 7):
        value = JOINT_SIGN[i] * float(home[offset + i - 1])
        data.qpos[joint(side, i)] = value
        data.ctrl[sim._actuator_ids[(side, i)]] = value
    grip = float(np.clip(home[offset + 6], 0.0, 0.07))
    data.qpos[joint(side, 7)] = grip / 2
    data.qpos[joint(side, 8)] = -grip / 2
    data.ctrl[sim._actuator_ids[(side, 7)]] = grip / 2
    data.ctrl[sim._actuator_ids[(side, 8)]] = -grip / 2
data.qpos[free:free + 7] = spawn(args.direction)
mujoco.mj_forward(model, data)


def state_vector(obs):
    return np.array([obs[f"{side}_{motor}.{part}"] for side in SIDES
                     for motor in MOTORS for part in ("pos", "effort")],
                    dtype=np.float32)


def action_dict(action14):
    return {f"{side}_{motor}.pos": float(action14[offset + index])
            for side, offset in (("left", 0), ("right", 7))
            for index, motor in enumerate(MOTORS)}


worker = DIRECTION[args.direction]["worker"]
print(f"{args.direction}: {worker} arm does the work")
print("frame  recorded                          predicted"
      "                           max|err|  grip")
worst = 0.0
for step in range(min(args.frames, len(stream_direction))):
    obs = sim.get_observation()
    observation = {"observation.state": state_vector(obs)}
    for name in CAMS:
        observation[f"observation.images.{name}"] = np.asarray(obs[name])
    pred = np.asarray(predict_action(
        observation=observation, policy=policy, device=torch.device("cuda"),
        preprocessor=pre, postprocessor=post, use_amp=cfg.use_amp,
        task=TASK, robot_type="piper_dual")).reshape(-1)
    clean = stream_direction[step]
    err = float(np.abs(pred[base:base + 6] - clean[base:base + 6]).max())
    worst = max(worst, err)
    if step % args.step == 0:
        cube_xyz = np.round(data.xpos[model.body("cube").id], 3)
        print(f"{step:5d}  {np.round(clean[base:base + 7], 2)}  "
              f"{np.round(pred[base:base + 7], 2)}  {err:7.3f}  cube {cube_xyz}")
    sim.send_action(action_dict(pred if args.closed_loop else clean))
print("worst error on the working arm: %.3f rad" % worst)
sim.disconnect()
