#!/usr/bin/env bash
# 双向循环（带 Rerun，快版）—— 外层先装好优化环境，再交给原样的 cube_loop_rerun.sh。
# 支持 --plan / --offline-check / 直接跑（会动机械臂）。
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
source "$HERE/fast_env.sh"
INNER="$HERE/../cube_loop_rerun.sh"
[ -f "$INNER" ] || INNER="${CUBE_ENVFIX_ROOT}/cube_loop_rerun.sh"
echo "[cube_rerun_fast] 内层脚本：$INNER"
exec bash "$INNER" "$@"
