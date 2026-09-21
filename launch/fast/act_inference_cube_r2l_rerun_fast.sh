#!/usr/bin/env bash
# 右→左 单次（带 Rerun，快版）—— 外层先装好优化环境，再交给原样的 *_rerun.sh。
# 优化内容与开关见本目录 README.txt。
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
INNER="$HERE/../act_inference_cube_r2l_rerun.sh"
[ -f "$INNER" ] || INNER="${CUBE_ENVFIX_ROOT}/act_inference_cube_r2l_rerun.sh"
echo "[cube_rerun_fast] 内层脚本：$INNER"
exec bash "$INNER" "$@"
