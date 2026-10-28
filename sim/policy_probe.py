"""Run the ACT checkpoint offline on the recorded demonstration.

Feeds the policy the exact observations the real robot saw, frame by frame, and
compares the predicted action with the action that was recorded. This separates
a policy/contract problem from a simulator-fidelity problem: if the prediction
tracks the demonstration here, any drift in the MuJoCo run comes from the
simulated observation.

Usage:  python policy_probe.py [--frames N] [--no-ensemble]
"""
import argparse
import glob

import numpy as np
import pandas as pd
import torch
import imageio.v2 as imageio

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.control_utils import predict_action

CKPT = ("/home/databall/VLA/lerobot_piper/outputs/train/"
        "cube_r2l_new_env_ft_from_80ep20k/checkpoints/020000/pretrained_model")
ROOT = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust")
TASK = "Place the cube to the other side."
np.set_printoptions(precision=3, suppress=True)

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--step", type=int, default=10,
                help="report every N frames")
ap.add_argument("--no-ensemble", action="store_true")
args = ap.parse_args()

df = pd.concat([pd.read_parquet(f)
                for f in sorted(glob.glob(f"{ROOT}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
states = np.stack(df["observation.state"].values).astype(np.float32)
actions = np.stack(df["action"].values)
videos = {name: imageio.get_reader(
    glob.glob(f"{ROOT}/videos/observation.images.{name}/*/*.mp4")[0])
    for name in ("left", "middle", "right")}
print("episode frames", len(states), "state dim", states.shape[1])

cfg = PreTrainedConfig.from_pretrained(CKPT)
# The saved config still points at the checkpoint it was fine-tuned from; the
# launcher overrides it with the path it was given, and so does this probe.
cfg.pretrained_path = CKPT
cfg.device = "cuda"
cfg.chunk_size = 100
cfg.n_action_steps = 1
cfg.n_obs_steps = 1
cfg.temporal_ensemble_coeff = None if args.no_ensemble else 0.01
policy = make_policy(cfg, ds_meta=LeRobotDatasetMetadata(
    "local/cube_r2l_dataset_v2_robust", root=ROOT))
pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=CKPT,
                                     dataset_stats=None)
policy.eval()
policy.reset()
pre.reset()
post.reset()

print("\nframe  recorded right arm                 predicted right arm"
      "                  max|err|  grip")
worst = 0.0
for f in range(min(args.frames, len(states))):
    obs = {"observation.state": states[f].copy()}
    for name, reader in videos.items():
        obs[f"observation.images.{name}"] = np.asarray(reader.get_data(f))
    pred = predict_action(observation=obs, policy=policy,
                          device=torch.device("cuda"), preprocessor=pre,
                          postprocessor=post, use_amp=cfg.use_amp, task=TASK,
                          robot_type="piper_dual")
    pred = np.asarray(pred).reshape(-1)
    err = float(np.abs(pred[7:13] - actions[f][7:13]).max())
    worst = max(worst, err)
    if f % args.step == 0:
        print(f"{f:5d}  {np.round(actions[f][7:13], 2)}  "
              f"{np.round(pred[7:13], 2)}  {err:8.3f}  {pred[13]:.3f}")
print("worst right-arm error over the run: %.3f rad" % worst)
