"""Drive the demonstration through the simulator and let the policy watch.

This is the same question as `policy_probe.py` with the observation source
swapped: the arm is driven by the recorded actions, so it is always exactly
where the demonstration was, and the policy is asked what it would do next
from the *simulated* images and state. If the predicted action tracks the
recorded one here, the simulated observation is good enough and a closed-loop
drift comes from feedback, not from the pictures. If it does not, the gap is in
the pictures or the state and can be measured directly.

Usage:  MUJOCO_GL=egl python policy_probe_sim.py [--frames N] [--step N]
"""
import argparse
import glob

import imageio.v2 as imageio

import numpy as np
import pandas as pd
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.control_utils import predict_action

from piper_dual_sim import PiperDualSim

CKPT = ("/home/databall/VLA/lerobot_piper/outputs/train/"
        "cube_r2l_new_env_ft_from_80ep20k/checkpoints/020000/pretrained_model")
ROOT = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust")
TASK = "Place the cube to the other side."
CAMS = ("left", "middle", "right")
MOTORS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
          "gripper")
SIDES = ("left", "right")
np.set_printoptions(precision=3, suppress=True)

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--step", type=int, default=10)
ap.add_argument("--real-cams", default="",
                help="comma separated camera names to feed from the recording")
ap.add_argument("--real-state", action="store_true",
                help="feed the recorded state instead of the simulated one; "
                     "positions are identical, the effort channel is not")
args = ap.parse_args()

df = pd.concat([pd.read_parquet(f)
                for f in sorted(glob.glob(f"{ROOT}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)
states = np.stack(df["observation.state"].values).astype(np.float32)
print("episode frames", len(actions))

cfg = PreTrainedConfig.from_pretrained(CKPT)
cfg.pretrained_path = CKPT          # the saved config names its parent checkpoint
cfg.device = "cuda"
cfg.chunk_size = 100
cfg.n_action_steps = 1
cfg.n_obs_steps = 1
cfg.temporal_ensemble_coeff = 0.01
policy = make_policy(cfg, ds_meta=LeRobotDatasetMetadata(
    "local/cube_r2l_dataset_v2_robust", root=ROOT))
pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=CKPT,
                                     dataset_stats=None)
policy.eval()

REAL_CAMS = [name for name in args.real_cams.split(",") if name]
videos = {name: imageio.get_reader(glob.glob(
    f"{ROOT}/videos/observation.images.{name}/*/*.mp4")[0])
    for name in REAL_CAMS}

sim = PiperDualSim(render=True)
sim.connect()


def observation_state(obs):
    return np.array([obs[f"{side}_{motor}.{part}"] for side in SIDES
                     for motor in MOTORS for part in ("pos", "effort")],
                    dtype=np.float32)


def action_dict(action14):
    return {f"{side}_{motor}.pos": float(action14[base + index])
            for side, base in (("left", 0), ("right", 7))
            for index, motor in enumerate(MOTORS)}


print("\nframe  recorded right arm                 predicted right arm"
      "                  max|err|  grip")
worst = 0.0
for f in range(min(args.frames, len(actions))):
    obs = sim.get_observation()
    state = states[f].copy() if args.real_state else observation_state(obs)
    observation = {"observation.state": state}
    for name in CAMS:
        observation[f"observation.images.{name}"] = (
            np.asarray(videos[name].get_data(f)) if name in videos
            else np.asarray(obs[name]))
    pred = np.asarray(predict_action(
        observation=observation, policy=policy, device=torch.device("cuda"),
        preprocessor=pre, postprocessor=post, use_amp=cfg.use_amp,
        task=TASK, robot_type="piper_dual")).reshape(-1)
    sim.send_action(action_dict(actions[f]))   # the demonstration drives the arm
    err = float(np.abs(pred[7:13] - actions[f][7:13]).max())
    worst = max(worst, err)
    if f % args.step == 0:
        print(f"{f:5d}  {np.round(actions[f][7:13], 2)}  "
              f"{np.round(pred[7:13], 2)}  {err:8.3f}  {pred[13]:.3f}")
sim.disconnect()
print("worst right-arm error over the run: %.3f rad" % worst)
