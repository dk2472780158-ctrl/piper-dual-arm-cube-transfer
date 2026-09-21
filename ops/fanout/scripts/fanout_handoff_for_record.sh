#!/usr/bin/env bash
# Stop idle fan-out daemon before lerobot-record; restart on exit.
# Source from run_piper_record.sh or: eval "$(bash fanout_handoff_for_record.sh --export)"
set -euo pipefail
# --- repository configuration (config/env.sh) -------------------------------
_piper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$_piper_dir" != "/" && ! -f "$_piper_dir/config/env.sh" ]]; do
    _piper_dir="$(dirname -- "$_piper_dir")"
done
# shellcheck source=/dev/null
source "$_piper_dir/config/env.sh"
unset _piper_dir

UNIT=piper-rgb-fanout.service
_RESTART_FANOUT=0

_stop_idle_fanout() {
  if systemctl is-active --quiet "${UNIT}" 2>/dev/null; then
    echo "[fanout] stopping ${UNIT} (release camera_*)"
    sudo systemctl stop "${UNIT}"
    _RESTART_FANOUT=1
    sleep 1
  fi
}

_start_idle_fanout() {
  if [[ "${_RESTART_FANOUT}" == "1" ]] && [[ "${PIPER_FANOUT_NO_RESTART:-0}" != "1" ]]; then
    echo "[fanout] restarting ${UNIT}"
    sudo systemctl start "${UNIT}" || true
  fi
}

if [[ "${1:-}" == "--export" ]]; then
  cat <<'EOF'
_stop_idle_fanout
trap _start_idle_fanout EXIT
EOF
  exit 0
fi

_stop_idle_fanout
trap _start_idle_fanout EXIT
exec "$@"
