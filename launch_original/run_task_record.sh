#!/usr/bin/env bash
set -euo pipefail
# --- repository configuration (config/env.sh) -------------------------------
_piper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$_piper_dir" != "/" && ! -f "$_piper_dir/config/env.sh" ]]; do
    _piper_dir="$(dirname -- "$_piper_dir")"
done
# shellcheck source=/dev/null
source "$_piper_dir/config/env.sh"
unset _piper_dir

source "${CONDA_SH}"
conda activate lerobot_v30

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

exec lerobot-record \
  --robot.type=piper_dual \
  --robot.left_port=can1 \
  --robot.right_port=can0 \
  --robot.read_only=true \
  --teleop.type=piper_dual_leader \
  --teleop.left_port=can1 \
  --teleop.right_port=can0 \
  --dataset.repo_id=local/cube_r2l_new_env_1 \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=90 \
  --dataset.fps=30 \
  --dataset.single_task="Place the cube to the other side." \
  --dataset.push_to_hub=false \
  --display_data=false \
  --show_control_window=true
