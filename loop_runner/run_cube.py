#!/usr/bin/env python3
"""Sequence existing ACT launchers. Never imports a robot driver itself."""
import argparse
import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
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
PROJECT = Path(
    os.environ.get("LEROBOT_REPO", os.path.expanduser("~/VLA/lerobot_piper"))
)
SCRIPTS = {
    'l2r': HERE / 'act_inference_cube_l2r_temporal.sh',
    'r2l': HERE / 'act_inference_cube_r2l_temporal.sh',
}


def sequence(mode, cycles):
    return ['l2r', 'r2l'] * cycles if mode == 'loop' else [mode]


def main():
    parser = argparse.ArgumentParser(description='Best ACT cube transfer; stop + Enter or Ctrl+C.')
    parser.add_argument('mode', choices=['loop', 'l2r', 'r2l'])
    parser.add_argument('--cycles', type=int, choices=range(1, 11), default=10)
    parser.add_argument('--plan', action='store_true', help='Print plan only; no hardware or child processes.')
    args = parser.parse_args()
    plan = sequence(args.mode, args.cycles)
    print('Plan: ' + ' -> '.join(plan), flush=True)
    print('Each direction: 15s policy rollout, then selected-arm reset.', flush=True)
    if args.plan:
        return 0
    # Serialize these new entry points; cannot lock unrelated legacy launchers.
    with LOCK_PATH.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another cube runner is active. Stop it first.')
            return 2
        return run(plan)


def run(plan):
    stop = threading.Event()
    child = None

    def request_stop(*_):
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)

    def read_commands():
        for line in sys.stdin:
            if line.strip().lower() in ('stop', 'q', 'quit', '停', '停止'):
                stop.set()
                break
        # Losing a supervising terminal must not start another motion.
        stop.set()

    if not sys.stdin.isatty():
        print('Run from an interactive terminal; unattended/background execution is disabled.')
        return 2
    threading.Thread(target=read_commands, daemon=True).start()
    run_dir = HERE / 'logs' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True)
    print(f'Logs: {run_dir}', flush=True)
    print('Starts now. Stop: type stop then Enter, or Ctrl+C. Keep hardware emergency stop available.', flush=True)
    completed = 0
    try:
        for index, direction in enumerate(plan, 1):
            if stop.is_set():
                break
            script = SCRIPTS[direction]
            if not script.is_file():
                raise RuntimeError(f'Missing launcher: {script}')
            log_path = run_dir / f'{index:02d}_{direction}.log'
            env = os.environ.copy()
            env.update(PYTHONUNBUFFERED='1', PIPER_ACTION_FILTER_ALPHA='1',
                       PIPER_CLIP_OBSERVATION_STATE='0')
            print(f'[{index}/{len(plan)}] {direction}; log={log_path}', flush=True)
            reset_completed = threading.Event()
            with log_path.open('w', encoding='utf-8') as log:
                child = subprocess.Popen(['bash', str(script)], cwd=PROJECT, env=env,
                                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, errors='replace',
                                         start_new_session=True)

                def forward_output(process):
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end='', flush=True)
                        if '[piper-auto-reset] complete' in line:
                            reset_completed.set()

                output = threading.Thread(target=forward_output, args=(child,), daemon=True)
                output.start()
                interrupted = False
                while child.poll() is None:
                    if stop.wait(0.1) and not interrupted:
                        interrupted = True
                        print('STOP requested: interrupting current stage; no next direction or extra reset.', flush=True)
                        try:
                            os.killpg(child.pid, signal.SIGINT)
                        except ProcessLookupError:
                            pass
                        # Do not SIGKILL a robot process or automatically start recovery motion.
                    if interrupted:
                        time.sleep(0.1)
                output.join()
                code = child.returncode
                child.stdout.close()
                child = None
            if stop.is_set():
                break
            if code != 0 or not reset_completed.is_set():
                raise RuntimeError(f'{direction} incomplete: exit={code}, reset={reset_completed.is_set()}; see {log_path}')
            completed += 1
            print(f'Completed directions={completed}; round trips={completed // 2}', flush=True)
    except Exception as error:
        stop.set()
        print(f'STOPPED: {error}', flush=True)
        return 1
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            child.wait()
        print(f'Finished: {completed}/{len(plan)} directions. No further motion scheduled.', flush=True)
    return 130 if stop.is_set() else 0


if __name__ == '__main__':
    raise SystemExit(main())
