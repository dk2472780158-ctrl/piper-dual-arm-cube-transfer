#!/usr/bin/env bash
# One-command dual-Piper reset. No interactive MOVE input is required.

set -o pipefail
# --- repository configuration (config/env.sh) -------------------------------
_piper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$_piper_dir" != "/" && ! -f "$_piper_dir/config/env.sh" ]]; do
    _piper_dir="$(dirname -- "$_piper_dir")"
done
# shellcheck source=/dev/null
source "$_piper_dir/config/env.sh"
unset _piper_dir

ACT_ROOT="${LEROBOT_REPO}"
RESET_SCRIPT="$ACT_ROOT/reset_piper_pose.py"
LOG_ROOT="${PIPER_WS}/experiments/piper_reset_logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_ROOT/reset_both_$RUN_ID.log"

mkdir -p "$LOG_ROOT"

trap 'echo; echo "收到停止请求；取消归位。"; exit 130' INT TERM

fail() {
    echo "ERROR: $1" >&2
    echo "归位日志：$LOG" >&2
    exit 1
}

echo "============================================================"
echo "双Piper一键归位"
echo "立即自动使能并移动，无需输入MOVE或等待倒计时"
echo "运行脚本前必须清空工作区、支撑机械臂并准备急停"
echo "============================================================"

[[ -f "$CONDA_SH" ]] || fail "找不到Conda初始化脚本：$CONDA_SH"
[[ -f "$RESET_SCRIPT" ]] || fail "找不到归位脚本：$RESET_SCRIPT"

if pgrep -af '[l]erobot-record|[r]obot_client|[a]ct_smolvla_hybrid_towel_safe.py' >/dev/null; then
    echo "检测到可能占用机械臂的进程：" >&2
    pgrep -af '[l]erobot-record|[r]obot_client|[a]ct_smolvla_hybrid_towel_safe.py' >&2
    fail "请先停止上面的机械臂控制进程"
fi

# shellcheck source=/dev/null
source "$CONDA_SH" || fail "Conda初始化失败"
conda activate "$LEROBOT_ENV" || fail "无法激活环境：$LEROBOT_ENV"
cd "$ACT_ROOT" || fail "无法进入ACT目录"

printf 'MOVE\n' | python -B "$RESET_SCRIPT" \
    --arm both \
    --left-can can1 \
    --right-can can0 \
    --execute \
    2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[1]}

if [[ "$STATUS" -ne 0 ]]; then
    fail "归位程序失败，状态码=$STATUS"
fi

if ! grep -q 'Pose reset complete' "$LOG"; then
    fail "没有检测到归位完成标志"
fi

ln -sfn "$LOG" "$LOG_ROOT/latest.log"

echo "============================================================"
echo "PIPER_RESET_SUCCESS"
echo "双臂已经归位并保持目标姿态"
echo "没有启动ACT或SmolVLA推理"
echo "日志：$LOG"
echo "============================================================"
