"""Watch the cell in MuJoCo's own window, with the policy driving it.

The rendered movies show one fixed view after the fact. This opens the live
viewer on the machine's display: the arms, the table, the mats and the cube are
all there, the camera can be dragged with the mouse, and the actions come from
the ACT checkpoint exactly as they do on the hardware.

    ./view.sh                 # the policy drives, 20 s
    ./view.sh demo            # the recorded demonstration drives instead
    ./view.sh policy 40       # 40 s of the policy
    ./view.sh --camera middle # look through the cell's own middle camera

Which checkpoint drives is a flag, not a constant. The default is the one
trained on the simulated cell (`cube_sim_act_v1_ft`, fine-tuned on episodes
recorded here); the checkpoint trained on the real recording cannot drive this
cell -- it sees the renders, not the photographs, and drops the cube. Pass
`--ckpt` to try another one:

    ./view.sh policy 40 --ckpt <path>/pretrained_model --ds-root <dataset>

The window has to land on a display; on this machine that is :0, where the
desktop session already runs.
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import mujoco
import mujoco.viewer

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.control_utils import predict_action

from directions import DIRECTION, MAT_RADIUS, MAT_TOP, mirror, spawn, stream
from piper_dual_sim import JOINT_SIGN, PiperDualSim

# The demonstration the `demo` mode replays, and the recording the real-cell
# checkpoint was trained on. Both stay the real cell: the mirrored left-to-right
# pass is that same recording reflected.
DEMO_ROOT = ("/home/databall/.cache/huggingface/lerobot/local/"
             "cube_r2l_dataset_v2_robust")
REAL_CKPT = ("/home/databall/VLA/lerobot_piper/outputs/train/"
             "cube_r2l_new_env_ft_from_80ep20k/checkpoints/020000/"
             "pretrained_model")
# The checkpoint that actually drives this cell: ACT fine-tuned on episodes
# recorded in the simulation itself.
SIM_CKPT = ("/home/databall/VLA/lerobot_piper/outputs/train/"
            "cube_sim_act_v1_ft/checkpoints/012000/pretrained_model")
SIM_ROOT = ("/home/databall/.cache/huggingface/lerobot/local/cube_sim_v1")
TASK = "Place the cube to the other side."
CAMS = ("left", "middle", "right")
MOTORS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
          "gripper")
SIDES = ("left", "right")

ap = argparse.ArgumentParser()
ap.add_argument("mode", nargs="?", default="policy", choices=("policy", "demo"))
ap.add_argument("seconds", nargs="?", type=float, default=0.0,
                help="stop after this many seconds; 0 keeps the window open "
                     "until it is closed")
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--direction", choices=("r2l", "l2r", "alternate"),
                default="alternate",
                help="which way the cube is carried each round")
ap.add_argument("--start", choices=("r2l", "l2r"), default="r2l",
                help="which direction an alternating run begins with")
ap.add_argument("--episode-frames", type=int, default=0,
                help="policy mode: frames per pass before the cube is re-parked "
                     "and the other arm takes over; 0 uses the demonstration "
                     "length, which is what the pass is known to fit in")
ap.add_argument("--once", action="store_true",
                help="run one pass and stop instead of looping")
ap.add_argument("--camera", default="free",
                choices=("free", "middle", "left", "right"),
                help="which camera the window looks through. 'free' is the "
                     "standing view of the whole cell and the only one the "
                     "mouse can move; the other three are the cell's own "
                     "cameras, the ones the policy sees")
ap.add_argument("--scene", default="scene_real.xml",
                help="scene file to run; scene_real.xml is the textured, "
                     "shadow-casting variant, scene.xml the flat original")
ap.add_argument("--ckpt", default=SIM_CKPT, help="ACT checkpoint to drive with")
ap.add_argument("--ds-root", default=SIM_ROOT,
                help="dataset whose metadata describes the policy's features")
args = ap.parse_args()

if args.mode == "policy" and args.ckpt == REAL_CKPT:
    # that checkpoint only ever saw the right-to-left pass
    args.direction = "r2l"

df = pd.concat([pd.read_parquet(f) for f in
                sorted(glob.glob(f"{DEMO_ROOT}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)
HOME = actions[0].copy()
STREAM = {name: stream(actions, name) for name in DIRECTION}

sim = PiperDualSim(scene=Path(__file__).resolve().parent / args.scene,
                   render=True)
sim.connect()
sim.get_observation()

policy = pre = post = cfg = None
if args.mode == "policy":
    print(f"loading the ACT checkpoint from {args.ckpt} ...")
    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.pretrained_path = args.ckpt
    cfg.device = "cuda"
    cfg.chunk_size = 100
    cfg.n_action_steps = 1
    cfg.n_obs_steps = 1
    cfg.temporal_ensemble_coeff = 0.01
    ds_repo_id = f"local/{os.path.basename(args.ds_root.rstrip('/'))}"
    policy = make_policy(cfg, ds_meta=LeRobotDatasetMetadata(
        ds_repo_id, root=args.ds_root))
    pre, post = make_pre_post_processors(policy_cfg=cfg,
                                         pretrained_path=args.ckpt,
                                         dataset_stats=None)
    policy.eval()


def observation_state(obs):
    return np.array([obs[f"{side}_{motor}.{part}"] for side in SIDES
                     for motor in MOTORS for part in ("pos", "effort")],
                    dtype=np.float32)


def action_dict(action14):
    return {f"{side}_{motor}.pos": float(action14[base + index])
            for side, base in (("left", 0), ("right", 7))
            for index, motor in enumerate(MOTORS)}


def qpos_of(name):
    return int(np.ravel(sim._model.joint(name).qposadr)[0])


def reset_cell(direction):
    """Put both arms on their start pose and the cube where this pass grabs it.

    The left-to-right pass is the right one reflected, so its start pose is the
    reflection of the recorded one; the cube takes the reflected offset too.
    """
    sim._reset_state(apply_ctrl=True)
    data, model = sim._data, sim._model
    home = HOME if direction == "r2l" else mirror(HOME)
    for side, base in (("left", 0), ("right", 7)):
        for index in range(1, 7):
            value = JOINT_SIGN[index] * float(home[base + index - 1])
            data.qpos[qpos_of(f"{side}_joint{index}")] = value
            data.ctrl[sim._actuator_ids[(side, index)]] = value
        opening = float(np.clip(home[base + 6], 0.0, 0.07))
        data.qpos[qpos_of(f"{side}_joint7")] = opening / 2
        data.qpos[qpos_of(f"{side}_joint8")] = -opening / 2
        data.ctrl[sim._actuator_ids[(side, 7)]] = opening / 2
        data.ctrl[sim._actuator_ids[(side, 8)]] = -opening / 2
    free = qpos_of("cube_free")
    data.qpos[free:free + 7] = spawn(direction)
    sim._mujoco.mj_forward(model, data)


cube = sim._model.body("cube").id
if args.seconds:
    print(f"{args.mode} mode, stopping after {args.seconds:.0f} s.")
else:
    print(f"{args.mode} mode, running until the window is closed.")

# How long one pass gets. The recording is the reference: the policy was tuned
# to finish inside that many frames, and a pass that needs longer has already
# gone wrong.
episode_frames = args.episode_frames or len(actions)


def report_pass(direction):
    """Score the pass the way loop.py does: cube in the mat, gripper open."""
    target = DIRECTION[direction]["target"]
    end = sim._data.xpos[cube].copy()
    offset = float(np.linalg.norm(end[:2] - np.asarray(target)))
    worker = "left" if direction == "l2r" else "right"
    held = float(sim._data.qpos[qpos_of(f"{worker}_joint7")]
                 - sim._data.qpos[qpos_of(f"{worker}_joint8")])
    ok = bool(offset < MAT_RADIUS and end[2] < MAT_TOP + 0.033 and held > 0.02)
    mat = "yellow" if direction == "r2l" else "purple"
    print(f"  {direction}: {'PASS' if ok else 'FAIL'}  cube {np.round(end, 3)}  "
          f"{100 * offset:4.1f} cm from the {mat} mat centre", flush=True)
    return ok


step = 0
round_index = 0
passed = []
direction = args.start if args.direction == "alternate" else args.direction
reset_cell(direction)
with mujoco.viewer.launch_passive(sim._model, sim._data) as viewer:
    if args.camera == "free":
        # A standing view of the whole cell; drag with the mouse once it is open.
        # The azimuth has to stay inside the room: the panels around the table
        # are opaque from behind, and a camera placed past them sees nothing.
        viewer.cam.lookat[:] = (0.0, 0.0, 0.80)
        viewer.cam.distance = 2.1
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -38.0
    else:
        # One of the cell's own cameras, held fixed: this is the view the
        # policy gets, so it is worth watching a pass through it.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = sim._model.camera(args.camera).id
    started = time.monotonic()
    # The loop is paced to the wall clock, not to the simulated clock: a step
    # costs more than a frame of real time here, so counting simulated frames
    # makes the window look like it ends early.
    while viewer.is_running() and (not args.seconds
                                   or time.monotonic() - started < args.seconds):
        loop_start = time.monotonic()
        obs = sim.get_observation()
        if args.mode == "policy":
            observation = {"observation.state": observation_state(obs)}
            for name in CAMS:
                observation[f"observation.images.{name}"] = np.asarray(obs[name])
            action = np.asarray(predict_action(
                observation=observation, policy=policy,
                device=torch.device("cuda"), preprocessor=pre,
                postprocessor=post, use_amp=cfg.use_amp, task=TASK,
                robot_type="piper_dual")).reshape(-1)
        else:
            action = STREAM[direction][min(step, len(actions) - 1)]
        sim.send_action(action_dict(action))
        viewer.sync()
        step += 1
        if step % int(args.fps) == 0:
            print(f"{time.monotonic() - started:5.1f}s  {direction}  "
                  f"grip R {action[13]:.3f}  "
                  f"cube {np.round(sim._data.xpos[cube], 3)}")
        if step >= episode_frames:
            # One pass is over. Park the cube where the next arm expects it and
            # hand over, so the cell runs the same round trip the real one does.
            passed.append(report_pass(direction))
            if args.once:
                break
            round_index += 1
            if args.direction == "alternate":
                direction = "l2r" if direction == "r2l" else "r2l"
            reset_cell(direction)
            if args.mode == "policy":
                # A chunk-queueing policy carries its leftovers into the next
                # pass: without this the first commands come from the end of the
                # previous one and the arm never leaves the mat it just left.
                policy.reset()
                for processor in (pre, post):
                    if hasattr(processor, "reset"):
                        processor.reset()
            step = 0
            print(f"{time.monotonic() - started:5.1f}s  round "
                  f"{round_index}, cube set for {direction}", flush=True)
        time.sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - loop_start)))

print("cube ended at", np.round(sim._data.xpos[cube], 4))
if passed:
    print(f"{sum(passed)}/{len(passed)} passes succeeded")
sim.disconnect()
