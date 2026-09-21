#!/usr/bin/env bash
set -e
# --- repository configuration (config/env.sh) -------------------------------
_piper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$_piper_dir" != "/" && ! -f "$_piper_dir/config/env.sh" ]]; do
    _piper_dir="$(dirname -- "$_piper_dir")"
done
# shellcheck source=/dev/null
source "$_piper_dir/config/env.sh"
unset _piper_dir

# ============================================================
# Piper Dual + 3x RealSense LeRobot Data Collection
# ============================================================

# ---------- 数据集 ----------
DATASET_NAME="piper_dataset_test_01"
DATASET_ROOT="${PIPER_WS}/datasets/${DATASET_NAME}"

# 采几条
NUM_EPISODES=3

# 每条最长录制时间，实际做完可按 → 提前结束
EPISODE_TIME=120

# 每次 reset 最长时间，摆好环境可按 → 提前结束
RESET_TIME=120

# 采集频率
FPS=30

# 任务描述
TASK="Piper dual arm manipulation task."

# ---------- Piper CAN ----------
# 已确认：
# can1 = 左侧
# can0 = 右侧
LEFT_PORT="can1"
RIGHT_PORT="can0"

# ---------- 项目 ----------
PROJECT_ROOT="${LEROBOT_REPO}"

cd "${PROJECT_ROOT}"

echo "============================================================"
echo " Piper Dual LeRobot Recording"
echo "============================================================"
echo "Dataset : ${DATASET_NAME}"
echo "Root    : ${DATASET_ROOT}"
echo "Episodes: ${NUM_EPISODES}"
echo "FPS     : ${FPS}"
echo "Task    : ${TASK}"
echo "Left CAN: ${LEFT_PORT}"
echo "RightCAN: ${RIGHT_PORT}"
echo "============================================================"

# 防止不小心覆盖已有正式数据
if [ -e "${DATASET_ROOT}" ]; then
    echo
    echo "[ERROR] Dataset already exists:"
    echo "        ${DATASET_ROOT}"
    echo
    echo "请修改 DATASET_NAME，或者确认是废弃数据后手动删除："
    echo "rm -rf '${DATASET_ROOT}'"
    exit 1
fi

mkdir -p ${PIPER_WS}/datasets

# shellcheck source=../ops/fanout/scripts/piper_lerobot_fanout_env.sh
source "${PIPER_FANOUT_ROOT}/scripts/piper_lerobot_fanout_env.sh"

python -m lerobot.scripts.lerobot_record \
  --robot.type=piper_dual \
  --robot.left_port="${LEFT_PORT}" \
  --robot.right_port="${RIGHT_PORT}" \
  --robot.read_only=true \
  --teleop.type=piper_dual_leader \
  --teleop.left_port="${LEFT_PORT}" \
  --teleop.right_port="${RIGHT_PORT}" \
  --dataset.repo_id="local/${DATASET_NAME}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME}" \
  --dataset.reset_time_s="${RESET_TIME}" \
  --dataset.fps="${FPS}" \
  --dataset.video=true \
  --dataset.single_task="${TASK}" \
  --dataset.push_to_hub=false \
  --display_data=false \
  --show_control_window=true \
  --play_sounds=false

echo
echo "============================================================"
echo " Recording finished"
echo " Dataset: ${DATASET_ROOT}"
echo "============================================================"
