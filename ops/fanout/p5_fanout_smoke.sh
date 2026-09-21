#!/usr/bin/env bash
# P5: open RealSense RGB (same nodes as lerobot-record) + fan-out to go2rtc.
# Requires ENABLE_PIPER_RGB_RELAY=1 on go2rtc-lab.
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

ROOT="${PIPER_FANOUT_ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export ENABLE_PIPER_RGB_PUBLISH=1
export PIPER_RGB_PUBLISH_CAMERAS="${PIPER_RGB_PUBLISH_CAMERAS:-all}"
export PIPER_RGB_PUBLISH_FPS="${PIPER_RGB_PUBLISH_FPS:-15}"

DURATION="${1:-45}"
exec ~/miniconda3/envs/lerobot_v30/bin/python "${ROOT}/fanout/p5_camera_fanout_smoke.py" --duration "${DURATION}"
