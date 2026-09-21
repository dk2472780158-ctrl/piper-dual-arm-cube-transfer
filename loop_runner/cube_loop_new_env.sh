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
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LEROBOT_REPO}"
ENV_ROOT="${LEROBOT_ENV}"
export PATH="$ENV_ROOT/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"
export PIPER_ACTION_FILTER_ALPHA=1
export PIPER_CLIP_OBSERVATION_STATE=0
export PYTHONUNBUFFERED=1
cd "$ROOT"
# Offline modes must not stop any camera services or otherwise touch hardware.
OFFLINE=0
for ARG in "$@"; do
    [[ "$ARG" == --plan || "$ARG" == --offline-check ]] && OFFLINE=1
done
if [[ "$OFFLINE" == 0 ]]; then
    source "${PIPER_FANOUT_ROOT}/scripts/piper_lerobot_fanout_env.sh"
fi
exec "$ENV_ROOT/bin/python" -B "$HERE/resident_new_env.py" loop "$@"
