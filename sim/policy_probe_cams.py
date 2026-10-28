"""Which camera's simulated view moves the policy's first action?

Starts from the recorded observation (state and all three images) and swaps in
one simulated camera at a time, then tries a couple of image treatments. The
recorded action is the reference, so the variant that stays near it is the one
the policy accepts.

Usage:  MUJOCO_GL=egl python policy_probe_cams.py
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
CAMS = ("left", "middle", "right")
np.set_printoptions(precision=3, suppress=True)

df = pd.concat([pd.read_parquet(f)
                for f in sorted(glob.glob(f"{ROOT}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
real_state = np.stack(df["observation.state"].values).astype(np.float32)[0]
real_action = np.stack(df["action"].values)[0]
real_images = {name: np.asarray(imageio.get_reader(
    glob.glob(f"{ROOT}/videos/observation.images.{name}/*/*.mp4")[0]).get_data(0))
    for name in CAMS}

sim = PiperDualSim(render=True)
sim.connect()
sim_obs = sim.get_observation()
sim.disconnect()
MOTORS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper")
sim_state = np.array([sim_obs[f"{s}_{m}.{p}"] for s in ("left", "right")
                      for m in MOTORS for p in ("pos", "effort")], dtype=np.float32)
sim_images = {name: np.asarray(sim_obs[name]) for name in CAMS}


def blur(image, sigma):
    if sigma <= 0:
        return image
    radius = int(3 * sigma)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    out = image.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        length = out.shape[axis]
        acc = np.zeros_like(out)
        for offset, weight in zip(offsets, kernel):
            index = [slice(None)] * out.ndim
            index[axis] = slice(offset + radius, offset + radius + length)
            acc += weight * padded[tuple(index)]
        out = acc
    return np.clip(out, 0, 255).astype(np.uint8)


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


def predict(images, state=real_state):
    policy.reset()
    pre.reset()
    post.reset()
    obs = {"observation.state": state.copy()}
    for name, image in images.items():
        obs[f"observation.images.{name}"] = image
    return np.asarray(predict_action(
        observation=obs, policy=policy, device=torch.device("cuda"),
        preprocessor=pre, postprocessor=post, use_amp=cfg.use_amp,
        task=TASK, robot_type="piper_dual")).reshape(-1)


print("recorded                      ", np.round(real_action[7:], 3))
variants = [("all real", real_images, False),
            ("all sim", sim_images, False),
            ("sim left", {**real_images, "left": sim_images["left"]}, False),
            ("sim middle", {**real_images, "middle": sim_images["middle"]}, False),
            ("sim right", {**real_images, "right": sim_images["right"]}, False),
            ("sim blur1.5", {k: blur(v, 1.5) for k, v in sim_images.items()}, False),
            ("sim blur3", {k: blur(v, 3.0) for k, v in sim_images.items()}, False),
            ("sim + sim state", sim_images, True)]
for name, images, use_sim_state in variants:
    state = sim_state if use_sim_state else real_state
    print(f"{name:16s}              ", np.round(predict(images, state)[7:], 3))
