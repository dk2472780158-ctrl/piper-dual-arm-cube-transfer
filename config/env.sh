#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Central configuration for piper-dual-arm-cube-transfer.
#
# Every launcher in this repository sources this file through the five-line
# preamble below, so no script contains a hard-coded machine path:
#
#     _dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
#     while [[ "$_dir" != "/" && ! -f "$_dir/config/env.sh" ]]; do
#         _dir="$(dirname -- "$_dir")"
#     done
#     source "$_dir/config/env.sh"
#
# Everything here is an ordinary shell variable with a sane default, so you can
# override any of it from your own shell before running a launcher:
#
#     PIPER_WS=~/robots/lab bash launch/cube_loop_rerun.sh
#
# ...or put persistent overrides in config/env.local.sh (git-ignored).
# ---------------------------------------------------------------------------

# Repository root, derived from this file's own location.
if [[ -z "${PIPER_REPO_ROOT:-}" ]]; then
    PIPER_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export PIPER_REPO_ROOT

# --- Workspace layout ------------------------------------------------------
# Root of the working tree that holds this project's *data*: the LeRobot
# checkout, datasets, checkpoints, Rerun recordings.  Kept separate from the
# repository so that code stays a clean clone and data stays out of git.
export PIPER_WS="${PIPER_WS:-$HOME/VLA}"

# Checkout of the LeRobot source tree.  Must contain src/lerobot/ and, once you
# have trained or downloaded a policy, outputs/train/<run>/checkpoints/...
export LEROBOT_REPO="${LEROBOT_REPO:-$PIPER_WS/lerobot_piper}"

# --- Python environment ----------------------------------------------------
# Conda *prefix* of the environment that runs everything.  The launchers only
# prepend "$LEROBOT_ENV/bin" to PATH; they never call `conda activate`, so this
# works the same from a login shell, a systemd unit or a container.
# If your environment lives somewhere else (a container often mounts it under a
# different prefix), point LEROBOT_ENV at wherever *your* interpreter is.
export LEROBOT_ENV="${LEROBOT_ENV:-$HOME/miniconda3/envs/lerobot_v30}"

# conda.sh, only needed by the few scripts that build a fresh environment.
export CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

# LeRobot's data home: cached datasets, normalisation statistics and local
# checkpoints are looked up here.  Dataset ids of the form `local/<name>`
# resolve to "$LEROBOT_HOME/local/<name>".
export LEROBOT_HOME="${LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"

# --- Component directories -------------------------------------------------
# Camera fan-out (the WebRTC/MPEG-TS publisher that owns the cameras while a
# task runs).  Shipped inside this repository, so the default just works.
export PIPER_FANOUT_ROOT="${PIPER_FANOUT_ROOT:-$PIPER_REPO_ROOT/ops/fanout}"

# Directory holding rerun_env.sh, i.e. where the launchers expect to find their
# environment helper.
export CUBE_ENVFIX_ROOT="${CUBE_ENVFIX_ROOT:-$PIPER_REPO_ROOT/launch}"

# The Rerun "fast" overlay (see docs/RERUN_PERFORMANCE.md).
export CUBE_RERUN_FAST_ROOT="${CUBE_RERUN_FAST_ROOT:-$PIPER_REPO_ROOT/launch/fast}"

# --- Rerun viewer ----------------------------------------------------------
# Where the Rerun gRPC server listens.  Set CUBE_RERUN_IP/PORT if you run the
# viewer on another host, or CUBE_RERUN_URL to pass a full URL instead.
export CUBE_RERUN_IP="${CUBE_RERUN_IP:-127.0.0.1}"
export CUBE_RERUN_PORT="${CUBE_RERUN_PORT:-9876}"

# Where .rrd recordings are written.  Deliberately *not* inside PIPER_WS's
# network-mounted subdirectories: a recording stream competes with the control
# link for upstream bandwidth, which shows up as jitter in the control loop.
export CUBE_RERUN_SAVE_DIR="${CUBE_RERUN_SAVE_DIR:-$PIPER_WS/rerun_data}"

# Where the Rerun SDK drops its bytecode cache.  Keeps a read-only SDK install
# from being re-compiled on every import (431 modules, tens of seconds).
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$PIPER_WS/.pycache}"

# --- Concurrency lock ------------------------------------------------------
# Every entry point that takes the CAN bus and the cameras serialises itself on
# one lock file, so an old and a new runner cannot come up at once and fight
# over the hardware.  Leave empty to let each runner derive the path from the
# repository root; set it only if you deploy the runners somewhere that has no
# config/env.sh above it.
export CUBE_RUNNER_LOCK="${CUBE_RUNNER_LOCK:-}"

# --- Optional third-party paths -------------------------------------------
# A vendored Rerun SDK that should shadow the one inside LEROBOT_ENV.  Leave
# empty to use whatever `import rerun` finds in the environment.
export RERUN_SDK_ROOT="${RERUN_SDK_ROOT:-}"

# A script that owns an idle camera preview (camera panel on an appliance-style
# deployment).  When set and executable it is stopped before a run takes the
# cameras and restarted afterwards.  Leave empty if you have no such helper.
export PIPER_IDLE_FANOUT_SH="${PIPER_IDLE_FANOUT_SH:-}"

# --- Local overrides -------------------------------------------------------
if [[ -f "$PIPER_REPO_ROOT/config/env.local.sh" ]]; then
    # shellcheck source=/dev/null
    source "$PIPER_REPO_ROOT/config/env.local.sh"
fi
