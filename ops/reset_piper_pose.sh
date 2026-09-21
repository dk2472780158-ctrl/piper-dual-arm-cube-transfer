#!/usr/bin/env bash
# 【会使机械臂运动】一键归位。速度与姿态都写死在下面，跑起来立刻使能并移动。
# 清空工作区、支撑好机械臂、急停随手可及之后再跑。
#
# 干跑（只读，不使能、不动）：
#     python ops/reset_piper_pose.py --arm both        # 去掉 --execute 就是干跑
set -euo pipefail
# --- repository configuration (config/env.sh) -------------------------------
_piper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$_piper_dir" != "/" && ! -f "$_piper_dir/config/env.sh" ]]; do
    _piper_dir="$(dirname -- "$_piper_dir")"
done
# shellcheck source=/dev/null
source "$_piper_dir/config/env.sh"
unset _piper_dir

cd "$(dirname -- "${BASH_SOURCE[0]}")"

python reset_piper_pose.py \
  --joint-speed 0.20 \
  --gripper-speed 0.01 \
  --arm both \
  --execute
