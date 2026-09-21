#!/usr/bin/env bash
# --- repository configuration (config/env.sh) -------------------------------
_piper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$_piper_dir" != "/" && ! -f "$_piper_dir/config/env.sh" ]]; do
    _piper_dir="$(dirname -- "$_piper_dir")"
done
# shellcheck source=/dev/null
source "$_piper_dir/config/env.sh"
unset _piper_dir
# piper_lerobot_fanout_env.sh — env for lerobot-record / inference → go2rtc hook
# Source from the launchers (launch/*.sh, launch_original/*.sh), never execute.
# Override: ENABLE_PIPER_RGB_PUBLISH=0
#
# 两块职责：
#   1) 把相机从「空闲预览」手里接过来 —— 一次只能有一个进程占着 camera_*；
#   2) 把 fanout 与 lerobot 源码放进 PYTHONPATH，并打开发布开关。
[[ "${BASH_SOURCE[0]}" == "${0}" ]] && {
  echo "source this file, do not execute" >&2
  exit 1
}

PROJECT_ROOT="${PROJECT_ROOT:-$LEROBOT_REPO}"
PIPER_FANOUT_ROOT="${PIPER_FANOUT_ROOT:-$PIPER_REPO_ROOT/ops/fanout}"

_piper_idle_fanout_restore() {
  if [[ -n "${PIPER_IDLE_FANOUT_SH:-}" && -x "${PIPER_IDLE_FANOUT_SH}" ]]; then
    bash "${PIPER_IDLE_FANOUT_SH}" start || true
  fi
}

# 把手持的 idle 预览停掉，腾出 camera_*。
# PIPER_IDLE_FANOUT_SH 为空 = 你的机器没有这类助手，直接跳过。
if [[ -n "${PIPER_IDLE_FANOUT_SH:-}" && -x "${PIPER_IDLE_FANOUT_SH}" ]]; then
  if ! bash "${PIPER_IDLE_FANOUT_SH}" stop; then
    echo "[fanout] ERROR: failed to stop idle fanout — cameras still busy" >&2
    return 1 2>/dev/null || exit 1
  fi
  # V4L 释放后稍等，避免 OpenCV 立刻 open 失败
  sleep 0.5
elif command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet piper-rgb-fanout.service 2>/dev/null; then
  echo "[fanout] stop piper-rgb-fanout.service (systemd)"
  sudo systemctl stop piper-rgb-fanout.service || true
fi

# 脚本/子 shell 结束时恢复 /videos 预览（只挂一次，避免重复 source 叠 trap）
if [[ -z "${_PIPER_FANOUT_IDLE_TRAP:-}" ]]; then
  export _PIPER_FANOUT_IDLE_TRAP=1
  trap '_piper_idle_fanout_restore' EXIT
fi

export PYTHONPATH="${PIPER_FANOUT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export ENABLE_PIPER_RGB_PUBLISH="${ENABLE_PIPER_RGB_PUBLISH:-1}"
export PIPER_RGB_PUBLISH_CAMERAS="${PIPER_RGB_PUBLISH_CAMERAS:-all}"
export PIPER_RGB_PUBLISH_FPS="${PIPER_RGB_PUBLISH_FPS:-15}"
