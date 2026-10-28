"""Is it the light? Vary nothing but the three pictures.

The earlier probes pin the drift on the images: hand the policy the recorded
frames and it tracks the demonstration, hand it the renders and it does not.
This probe narrows the pixels down. Every variant replays the same
demonstration through the same physics and feeds the policy the same simulated
state; only the three camera images change. Variants come in pairs, so the
answer is readable either way:

    sim            the shipped pipeline (render plus the fitted gain field)
    sim_nogain     the raw render, to show what the gain field is doing
    sim_photo      the shipped render with its mean and spread matched to the
                   recorded frame -- brightness, white balance and contrast
                   only, no lighting change
    shadow         the two directional lights allowed to cast shadows
    panel          shadows plus a ceiling panel of three wide spots
    panel_photo    the panel rig, photometrically matched
    real           the recorded frames, the reference
    real_to_sim    the recorded frames pushed onto the render's colour
                   statistics -- if the policy is photometrically brittle this
                   is what happens to good pictures that are merely mis-toned

If `sim_photo` recovers the reference the gap is tone. If only `panel_photo`
does, it is the light rig -- shadow structure and fall-off. If neither comes
close, the pictures differ in what they contain rather than how they are lit.

    MUJOCO_GL=egl python policy_probe_light.py --frames 441 --stride 3
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import imageio.v2 as imageio
import mujoco

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.control_utils import predict_action

import render_sensor
from piper_dual_sim import PiperDualSim

HERE = Path(__file__).resolve().parent
SCENE = HERE / "scene.xml"
CKPT = ("/home/databall/VLA/lerobot_piper/outputs/train/"
        "cube_r2l_new_env_ft_from_80ep20k/checkpoints/020000/pretrained_model")
ROOT = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust")
TASK = "Place the cube to the other side."
CAMS = ("left", "middle", "right")
MOTORS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
          "gripper")
SIDES = ("left", "right")
HEIGHT, WIDTH = 480, 640

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=441)
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--variants", default="sim,sim_nogain,sim_photo,shadow,panel,"
                                     "panel_photo,real,real_to_sim")
args = ap.parse_args()

df = pd.concat([pd.read_parquet(f)
                for f in sorted(glob.glob(f"{ROOT}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)
frames = min(args.frames, len(actions))
print("episode frames", len(actions), "evaluating", frames,
      "every", args.stride)

# --- the light rigs, as scene files ------------------------------------------
def light_model(kind):
    """scene.xml with a different set of lights; geometry is untouched.

    Lights live in the worldbody and own no joint, so the model's qpos layout
    is identical to the shipped one and a state can be copied straight across.
    """
    text = SCENE.read_text()
    match = re.search(r'meshdir="([^"]+)"', text)
    # from_xml_string resolves meshes from the working directory, not the scene
    text = text.replace(match.group(0),
                        f'meshdir="{(HERE / match.group(1)).resolve()}"')
    text = text.replace('castshadow="false"', 'castshadow="true"')
    if kind == "panel":
        panel = "\n".join(
            f'    <light name="panel_{i}" pos="-0.22 {y:.2f} 1.90" dir="0 0 -1" '
            f'directional="false" cutoff="85" exponent="1" '
            f'diffuse="0.40 0.40 0.40" ambient="0.06 0.06 0.06" '
            f'specular="0.03 0.03 0.03" castshadow="true"/>'
            for i, y in enumerate((-0.30, 0.0, 0.30)))
        text = text.replace("  </worldbody>", panel + "\n  </worldbody>")
    model = mujoco.MjModel.from_xml_string(text)
    assert model.nq == mujoco.MjModel.from_xml_path(str(SCENE)).nq
    return model


def photometric(source, target):
    """Give `source` the per-channel mean and spread of `target`."""
    out = {}
    for name in source:
        a = source[name].astype(np.float32)
        b = target[name].astype(np.float32)
        mean_a, std_a = a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0)
        mean_b, std_b = b.reshape(-1, 3).mean(0), b.reshape(-1, 3).std(0)
        out[name] = np.clip((a - mean_a) * (std_b / np.maximum(std_a, 1e-3))
                            + mean_b, 0, 255).astype(np.uint8)
    return out


def image_distance(images, reference):
    return {name: float(np.abs(images[name].astype(np.int16)
                               - reference[name].astype(np.int16)).mean())
            for name in CAMS}


# --- the policy ---------------------------------------------------------------
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

gain_model = render_sensor.load()


def observation_state(obs):
    return np.array([obs[f"{side}_{motor}.{part}"] for side in SIDES
                     for motor in MOTORS for part in ("pos", "effort")],
                    dtype=np.float32)


def action_dict(action14):
    return {f"{side}_{motor}.pos": float(action14[base + index])
            for side, base in (("left", 0), ("right", 7))
            for index, motor in enumerate(MOTORS)}


def run_pass(variant):
    """Replay the demonstration once with this variant's pictures."""
    sim = PiperDualSim(render=True)
    sim.connect()
    sim._render_enabled = False        # the frames are rendered below, on purpose
    base_cams = {name: mujoco.Renderer(sim._model, HEIGHT, WIDTH) for name in CAMS}
    alt = None
    if variant in ("shadow", "panel", "panel_photo"):
        model = light_model("panel" if variant.startswith("panel") else "shadow")
        alt = ({name: mujoco.Renderer(model, HEIGHT, WIDTH) for name in CAMS},
               model, mujoco.MjData(model))
    need_render = variant != "real"
    readers = {name: imageio.get_reader(glob.glob(
        f"{ROOT}/videos/observation.images.{name}/*/*.mp4")[0]) for name in CAMS}

    policy.reset()
    pre.reset()
    post.reset()
    errors, grip_errors, distances = [], [], {name: [] for name in CAMS}
    for f in range(frames):
        obs = sim.get_observation()
        real = {name: np.asarray(readers[name].get_data(f)) for name in CAMS}
        raw = {}
        if need_render:
            for name in CAMS:
                base_cams[name].update_scene(sim._data, camera=name)
                raw[name] = base_cams[name].render()[:, :, :3]
        alt_raw = {}
        if alt is not None:
            renderers, model, data = alt
            data.qpos[:] = sim._data.qpos
            mujoco.mj_forward(model, data)
            for name in CAMS:
                renderers[name].update_scene(data, camera=name)
                alt_raw[name] = renderers[name].render()[:, :, :3]

        if variant == "sim":
            images = {name: render_sensor.apply(gain_model, name, raw[name])
                      for name in CAMS}
        elif variant == "sim_nogain":
            images = raw
        elif variant == "sim_photo":
            gained = {name: render_sensor.apply(gain_model, name, raw[name])
                      for name in CAMS}
            images = photometric(gained, real)
        elif variant in ("shadow", "panel"):
            images = alt_raw
        elif variant == "panel_photo":
            images = photometric(alt_raw, real)
        elif variant == "real":
            images = real
        elif variant == "real_to_sim":
            gained = {name: render_sensor.apply(gain_model, name, raw[name])
                      for name in CAMS}
            images = photometric(real, gained)
        else:
            raise SystemExit(f"unknown variant {variant}")

        observation = {"observation.state": observation_state(obs)}
        for name in CAMS:
            observation[f"observation.images.{name}"] = images[name]
        pred = np.asarray(predict_action(
            observation=observation, policy=policy, device=torch.device("cuda"),
            preprocessor=pre, postprocessor=post, use_amp=cfg.use_amp,
            task=TASK, robot_type="piper_dual")).reshape(-1)
        sim.send_action(action_dict(actions[f]))

        if f % args.stride == 0:
            errors.append(float(np.abs(pred[7:13] - actions[f][7:13]).max()))
            grip_errors.append(float(abs(pred[13] - actions[f][13])))
            for name, value in image_distance(images, real).items():
                distances[name].append(value)

    for renderer in base_cams.values():
        renderer.close()
    if alt is not None:
        for renderer in alt[0].values():
            renderer.close()
    for reader in readers.values():
        reader.close()
    sim.disconnect()
    return (np.asarray(errors), np.asarray(grip_errors),
            {name: float(np.mean(v)) for name, v in distances.items()})


wanted = [name for name in args.variants.split(",") if name]
print("\n  recorded right arm over the run: "
      f"|a| max {np.abs(actions[:frames, 7:13]).max():.2f} rad\n")
print("variant        worst   mean   grip   | image L1 to real  L/M/R   "
      "gap closed")
results = {}
for variant in wanted:
    errors, grip, distance = run_pass(variant)
    results[variant] = (errors, grip, distance)
    print(f"{variant:12s}  {errors.max():6.3f} {errors.mean():6.3f} "
          f"{grip.mean():6.3f}   | "
          f"{distance['left']:5.1f} {distance['middle']:5.1f} "
          f"{distance['right']:5.1f}", flush=True)

if "sim" in results and "real" in results:
    base = results["sim"][0].max()
    target = results["real"][0].max()
    print(f"\ngap closed = (sim {base:.3f} - variant) / (sim - real "
          f"{target:.3f})")
    for variant, (errors, _, _) in results.items():
        closed = (base - errors.max()) / max(base - target, 1e-6)
        print(f"  {variant:12s} {100 * closed:6.1f}%   worst {errors.max():.3f}")
