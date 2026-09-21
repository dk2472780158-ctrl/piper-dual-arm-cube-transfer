#!/usr/bin/env python3
"""Persistent ACT pair; reuse original record_loop/reset, never reconnect between legs."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

HERE = Path(__file__).resolve().parent


def _lock_path() -> Path:
    """One lock file for every entry point that takes the CAN bus and cameras.

    It is resolved relative to the repository root, so the launchers under
    launch/ and the ones in this directory contend for the *same* file.  If
    this script has been copied out of the repository there is no config/env.sh
    above it, and it falls back to the directory it lives in.
    """
    for parent in (HERE, *HERE.parents):
        if (parent / 'config' / 'env.sh').is_file():
            return parent / 'loop_runner' / '.run.lock'
    return HERE / '.run.lock'


LOCK_PATH = Path(os.environ.get('CUBE_RUNNER_LOCK') or _lock_path())
ROOT = Path(
    os.environ.get("LEROBOT_REPO", os.path.expanduser("~/VLA/lerobot_piper"))
)
CHECKPOINTS = {
    'l2r': ROOT / 'outputs/train/cube_l2r_left_arm_act_v1_robust_c100/checkpoints/050000/pretrained_model',
    'r2l': ROOT / 'outputs/train/cube_r2l_new_env_ft_from_80ep20k/checkpoints/020000/pretrained_model',
}


def plan(mode, cycles):
    return ['l2r', 'r2l'] * cycles if mode == 'loop' else [mode]


def execute_sequence(directions, rollout, reset, announce):
    for index, direction in enumerate(directions, 1):
        announce(index, direction)
        rollout(direction)
        reset('left' if direction == 'l2r' else 'right')


def features():
    motors = [f'{side}_{name}' for side in ('left', 'right')
              for name in [*(f'joint_{i}' for i in range(1, 7)), 'gripper']]
    result = {
        'action': {'dtype': 'float32', 'shape': (14,), 'names': [m + '.pos' for m in motors]},
        'observation.state': {'dtype': 'float32', 'shape': (28,),
                              'names': [m + '.' + k for m in motors for k in ('pos', 'effort')]},
    }
    for name in ('left', 'right', 'middle'):
        result[f'observation.images.{name}'] = {
            'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels']}
    return result


class NumericLog:
    fps = 30

    def __init__(self, schema):
        self.features = schema
        self.rows = []

    def add_frame(self, frame):
        self.rows.append({'time': time.time(),
                          'state': frame['observation.state'].tolist(),
                          'action': frame['action'].tolist()})


def load_models(core, schema):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    models = {}
    for direction, path in CHECKPOINTS.items():
        cfg = PreTrainedConfig.from_pretrained(str(path))
        cfg.device = 'cuda'
        cfg.chunk_size = 100
        cfg.n_action_steps = 1
        cfg.temporal_ensemble_coeff = 0.01
        cfg.pretrained_path = str(path)
        assert cfg.input_features['observation.state'].shape == (28,)
        assert cfg.output_features['action'].shape == (14,)
        policy = ACTPolicy.from_pretrained(str(path), config=cfg).to('cuda').eval()
        pre, post = core.make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=str(path),
            preprocessor_overrides={'device_processor': {'device': 'cuda'},
                                    'rename_observations_processor': {'rename_map': {}}})
        models[direction] = policy, pre, post
        print(f'MODEL_READY {direction}: {path}', flush=True)
    return models


def warmup(core, models, schema):
    import numpy as np
    import torch
    from lerobot.utils.control_utils import predict_action
    obs = {'observation.state': np.zeros(28, dtype=np.float32)}
    obs.update({key: np.zeros((480, 640, 3), dtype=np.uint8)
                for key in schema if key.startswith('observation.images.')})
    for direction, (policy, pre, post) in models.items():
        for _ in range(2):
            action = predict_action(obs, policy, torch.device('cuda'), pre, post,
                                    policy.config.use_amp, task='Place the cube to the other side.',
                                    robot_type='piper_dual')
            if tuple(action.shape) != (1, 14) and tuple(action.shape) != (14,):
                raise RuntimeError(f'Unexpected action shape: {action.shape}')
            if not torch.isfinite(action).all():
                raise RuntimeError('Nonfinite warmup output')
        policy.reset()
        pre.reset()
        post.reset()
        print(f'WARMUP_OK {direction}; synthetic observations only, no hardware', flush=True)
    print(f'GPU allocated GiB: {torch.cuda.memory_allocated() / 2**30:.3f}', flush=True)


def offline_loop_check(core, models, schema):
    """Exercise original control code against a fake robot; no SDK instance."""
    import numpy as np

    class FakeRobot:
        robot_type = 'piper_dual'
        sent = 0

        def get_observation(self):
            obs = {name: 0.0 for name in schema['observation.state']['names']}
            obs.update({name: np.zeros((480, 640, 3), dtype=np.uint8)
                        for name in ('left', 'middle', 'right')})
            return obs

        def send_action(self, action):
            if set(action) != set(schema['action']['names']):
                raise RuntimeError('Action names mismatch')
            self.sent += 1
            return action

    fake = FakeRobot()
    teleop, action_proc, obs_proc = core.make_default_processors()
    for direction, (policy, pre, post) in models.items():
        log = NumericLog(schema)
        core.record_loop(robot=fake,
            events={'exit_early': False, 'stop_recording': False, 'rerecord_episode': False},
            fps=30, teleop_action_processor=teleop, robot_action_processor=action_proc,
            robot_observation_processor=obs_proc, dataset=log,
            policy=policy, preprocessor=pre, postprocessor=post, control_time_s=0.1,
            single_task='Place the cube to the other side.', display_data=False)
        assert log.rows
        print(f'FAKE_ROBOT_LOOP_OK {direction}; rows={len(log.rows)}', flush=True)


def camera_index(name):
    device = Path('/dev/camera_' + name)
    resolved = device.resolve(strict=True)
    if resolved.name.startswith('video') and resolved.name[5:].isdigit():
        return int(resolved.name[5:])
    for other in Path('/dev').glob('video*'):
        if other.stat().st_rdev == device.stat().st_rdev:
            return int(other.name[5:])
    raise RuntimeError(f'Cannot resolve camera: {device}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['loop', 'l2r', 'r2l'], nargs='?', default='loop')
    parser.add_argument('--cycles', type=int, choices=range(1, 11), default=10)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--offline-check', action='store_true')
    args = parser.parse_args()
    directions = plan(args.mode, args.cycles)
    print('RESIDENT plan:', directions, flush=True)
    if args.plan:
        return 0
    # Same lock as every other entry point: prevents an old and a new runner
    # from coming up at once and fighting over the CAN bus and the cameras.
    with LOCK_PATH.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another cube runner is active')
        return run(args, directions)


def run(args, directions):
    import lerobot.scripts.piper_cube_record as core
    schema = features()
    models = load_models(core, schema)
    warmup(core, models, schema)
    if args.offline_check:
        offline_loop_check(core, models, schema)
        print('OFFLINE_CHECK_PASS; no robot instantiated, no cameras/CAN opened', flush=True)
        return 0
    if not sys.stdin.isatty():
        raise RuntimeError('Interactive terminal required')

    def stop(*_):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)

    def commands():
        for line in sys.stdin:
            if line.strip().lower() in ('stop', 'q', 'quit', '停', '停止'):
                break
        os.kill(os.getpid(), signal.SIGINT)

    from lerobot.robots.piper_dual.config_piper_dual import PIPERDualConfig
    from lerobot.robots.piper_dual.piper_dual import PIPERDual
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    cameras = {name: OpenCVCameraConfig(index_or_path=camera_index(name),
                                      width=640, height=480, fps=30)
               for name in ('left', 'middle', 'right')}
    robot = None
    run_dir = HERE / 'resident_new_env_logs' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True)
    (run_dir / 'run_config.json').write_text(json.dumps({
        'checkpoints': {k: str(v) for k, v in CHECKPOINTS.items()},
        'directions': directions, 'seconds_per_direction': 15, 'fps_target': 30,
        'temporal_ensemble_coeff': 0.01, 'camera_indices': {
            name: cfg.index_or_path for name, cfg in cameras.items()}}, indent=2))
    active_log = None
    stage = 0
    try:
        threading.Thread(target=commands, daemon=True).start()
        robot = PIPERDual(PIPERDualConfig(left_port='can1', right_port='can0',
                                         read_only=False, cameras=cameras))
        # Verify names/order using the same feature builders as the original runner.
        teleop_proc, action_proc, obs_proc = core.make_default_processors()
        actual = core.combine_feature_dicts(
            core.aggregate_pipeline_dataset_features(pipeline=teleop_proc,
                initial_features=core.create_initial_features(action=robot.action_features), use_videos=True),
            core.aggregate_pipeline_dataset_features(pipeline=obs_proc,
                initial_features=core.create_initial_features(observation=robot.observation_features), use_videos=True))
        for key in ('action', 'observation.state'):
            if list(actual[key]['names']) != schema[key]['names']:
                raise RuntimeError(f'Feature order mismatch: {key}')
        robot.connect()
        print('CONNECTED_ONCE. Starts now; stop + Enter or Ctrl+C. No automatic reset on interruption.', flush=True)

        def announce(index, direction):
            nonlocal stage
            stage = index
            print(f'STAGE {index}/{len(directions)} {direction}', flush=True)

        def rollout(direction):
            nonlocal active_log
            active_log = NumericLog(actual)
            policy, pre, post = models[direction]
            # Original function resets the policy/Temporal history and both processors.
            core.record_loop(robot=robot,
                events={'exit_early': False, 'stop_recording': False, 'rerecord_episode': False},
                fps=30, teleop_action_processor=teleop_proc, robot_action_processor=action_proc,
                robot_observation_processor=obs_proc, dataset=active_log,
                policy=policy, preprocessor=pre, postprocessor=post, control_time_s=15,
                single_task='Place the cube to the other side.', display_data=False)
            save_log(direction)

        def save_log(direction):
            nonlocal active_log
            if active_log is not None:
                with (run_dir / f'{stage:02d}_{direction}.jsonl').open('w') as out:
                    for row in active_log.rows:
                        out.write(json.dumps(row) + '\n')
                active_log = None

        def reset(arm):
            started = time.perf_counter()
            core.reset_connected_piper(robot, arm=arm, joint_speed=0.8, gripper_speed=0.01,
                fps=30, settle_timeout=8, settle_tolerance=0.02, gripper_tracking_limit=0.01)
            print(f'RESET_DONE {arm} seconds={time.perf_counter()-started:.3f}; next model already loaded', flush=True)

        execute_sequence(directions, rollout, reset, announce)
        print('RESIDENT_COMPLETE', flush=True)
        return 0
    except KeyboardInterrupt:
        print('STOPPED: no next stage, no extra return-home motion. Hardware may remain enabled.', flush=True)
        return 130
    finally:
        if robot is not None:
            robot.disconnect()
        if active_log is not None:
            with (run_dir / f'{stage:02d}_interrupted.jsonl').open('w') as out:
                for row in active_log.rows:
                    out.write(json.dumps(row) + '\n')
        print(f'Logs: {run_dir}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
