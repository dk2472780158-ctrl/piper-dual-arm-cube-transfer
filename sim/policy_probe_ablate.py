"""Which channel of the simulated observation pushes the policy off course?

Builds the frame-0 observation from mixtures of real and simulated position,
effort and images, and prints the predicted right-arm action for each. The
recorded action is the reference: whatever mixture lands near it is the
observation the policy expects.

Usage:  MUJOCO_GL=egl python policy_probe_ablate.py
"""
import glob

import numpy as np
import pandas as pd
import torch
import imageio.v2 as imageio

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
SIDES = ("left", "right")
MOTORS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper")
np.set_printoptions(precision=3, suppress=True)

df = pd.concat([pd.read_parquet(f)
                for f in sorted(glob.glob(f"{ROOT}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
real_state = np.stack(df["observation.state"].values).astype(np.float32)[0]
real_action = np.stack(df["action"].values)[0]
real_images = {name: np.asarray(imageio.get_reader(
    glob.glob(f"{ROOT}/videos/observation.images.{name}/*/*.mp4")[0]).get_data(0))
    for name in ("left", "middle", "right")}

sim = PiperDualSim(render=True)
sim.connect()
sim_obs = sim.get_observation()
sim.disconnect()
sim_state = np.array([sim_obs[f"{s}_{m}.{p}"] for s in SIDES for m in MOTORS
                      for p in ("pos", "effort")], dtype=np.float32)
sim_images = {name: np.asarray(sim_obs[name]) for name in ("left", "middle", "right")}
imageio.imwrite("/tmp/ablate_sim_middle.png", sim_images["middle"])

cfg = PreTrainedConfig.from_pretrained(CKPT)
cfg.pretrained_path = CKPT
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


def photometric(source, target):
    """Per-channel affine that gives `source` the colour statistics of `target`."""
    out = {}
    for name in source:
        a = source[name].astype(np.float32)
        b = target[name].astype(np.float32)
        mean_a, std_a = a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0)
        mean_b, std_b = b.reshape(-1, 3).mean(0), b.reshape(-1, 3).std(0)
        scale = std_b / np.maximum(std_a, 1e-3)
        out[name] = np.clip((a - mean_a) * scale + mean_b, 0, 255).astype(np.uint8)
    return out


matched_images = photometric(sim_images, real_images)


def mix(position, effort, images):
    """State interleaves pos and effort per motor, so they mix per index."""
    state = np.empty(28, np.float32)
    state[0::2] = real_state[0::2] if position == "real" else sim_state[0::2]
    state[1::2] = real_state[1::2] if effort == "real" else sim_state[1::2]
    frames = {"real": real_images, "sim": sim_images, "matched": matched_images}
    return state, frames[images]


print("  position effort images | predicted right arm (j1..j6, gripper)")
print("  recorded              |", np.round(real_action[7:], 3))
for position in ("real", "sim"):
    for effort in ("real", "sim"):
        for images in ("real", "sim", "matched"):
            state, frames = mix(position, effort, images)
            policy.reset()
            pre.reset()
            post.reset()
            obs = {"observation.state": state.copy()}
            for name, image in frames.items():
                obs[f"observation.images.{name}"] = image
            pred = np.asarray(predict_action(
                observation=obs, policy=policy, device=torch.device("cuda"),
                preprocessor=pre, postprocessor=post, use_amp=cfg.use_amp,
                task=TASK, robot_type="piper_dual")).reshape(-1)
            print(f"  {position:8s} {effort:6s} {images:6s} | {np.round(pred[7:], 3)}")
