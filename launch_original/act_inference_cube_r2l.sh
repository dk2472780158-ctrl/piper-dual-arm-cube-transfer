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

PROJECT_ROOT="${LEROBOT_REPO}"
cd "$PROJECT_ROOT"

# 同时加入项目根目录和 src：
# - src 用于导入 lerobot
# - 项目根目录用于导入 reset_piper_pose.py
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"

# 停止空闲相机服务，并允许推理期间向 WebRTC 发布画面
# shellcheck source=../ops/fanout/scripts/piper_lerobot_fanout_env.sh
source "${PIPER_FANOUT_ROOT}/scripts/piper_lerobot_fanout_env.sh"

# CHECKPOINT="$PROJECT_ROOT/outputs/train/cube_r2l_act_v2_robust_c100/checkpoints/050000/pretrained_model"
CHECKPOINT="$PROJECT_ROOT/outputs/train/cube_r2l_act_v3_80ep_ft20000/checkpoints/020000/pretrained_model"

if [[ ! -f "$CHECKPOINT/config.json" ]]; then
    echo "ERROR: checkpoint not found:"
    echo "$CHECKPOINT"
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/src/lerobot/scripts/piper_cube_record.py" ]]; then
    echo "ERROR: dedicated inference module not found:"
    echo "$PROJECT_ROOT/src/lerobot/scripts/piper_cube_record.py"
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/reset_piper_pose.py" ]]; then
    echo "ERROR: reset module not found:"
    echo "$PROJECT_ROOT/reset_piper_pose.py"
    exit 1
fi

# 0.8 = 80% 使用新动作，20% 保留上一动作
# 设置为 1 可以完全关闭额外动作滤波
export PIPER_ACTION_FILTER_ALPHA="${PIPER_ACTION_FILTER_ALPHA:-0.2}"

# 新模型默认关闭临时 state 裁剪，直接使用稳健归一化
export PIPER_CLIP_OBSERVATION_STATE="${PIPER_CLIP_OBSERVATION_STATE:-0}"

# 只有手动开启临时裁剪时才会使用此文件
export PIPER_STATE_STATS_PATH="${PIPER_STATE_STATS_PATH:-${LEROBOT_HOME}/local/cube_r2l_dataset_v2_robust/meta/stats.json}"

LEFT_DEV="$(readlink -f /dev/camera_left || true)"
MIDDLE_DEV="$(readlink -f /dev/camera_middle || true)"
RIGHT_DEV="$(readlink -f /dev/camera_right || true)"

for CAMERA_DEV in "$LEFT_DEV" "$MIDDLE_DEV" "$RIGHT_DEV"; do
    if [[ ! "$CAMERA_DEV" =~ ^/dev/video[0-9]+$ ]]; then
        echo "ERROR: invalid camera device: ${CAMERA_DEV:-missing}" >&2
        exit 1
    fi

    if [[ ! -e "$CAMERA_DEV" ]]; then
        echo "ERROR: camera device does not exist: $CAMERA_DEV" >&2
        exit 1
    fi
done

LEFT_IDX="${LEFT_DEV#/dev/video}"
MIDDLE_IDX="${MIDDLE_DEV#/dev/video}"
RIGHT_IDX="${RIGHT_DEV#/dev/video}"

RUN_NAME="eval_cube_r2l_050000_$(date +%Y%m%d_%H%M%S)"
DATASET_ID="local/${RUN_NAME}"

echo "left=$LEFT_DEV"
echo "middle=$MIDDLE_DEV"
echo "right=$RIGHT_DEV"
echo "fanout ENABLE_PIPER_RGB_PUBLISH=${ENABLE_PIPER_RGB_PUBLISH:-unset}"
echo "checkpoint=$CHECKPOINT"
echo "result=$DATASET_ID"
echo "filter alpha=$PIPER_ACTION_FILTER_ALPHA"
echo "state clipping=$PIPER_CLIP_OBSERVATION_STATE"
echo "automatic reset arm=right"
echo
echo "WARNING: Clear the workspace and keep the emergency stop ready."
echo

python -m lerobot.scripts.piper_cube_record \
    --auto_reset_arm=right \
    --reset_joint_speed=0.5 \
    --reset_gripper_speed=0.01 \
    --reset_fps=30 \
    --reset_settle_timeout=8 \
    --reset_settle_tolerance=0.02 \
    --reset_gripper_tracking_limit=0.01 \
    --robot.type=piper_dual \
    --robot.left_port=can1 \
    --robot.right_port=can0 \
    --robot.cameras="{\"left\":{\"type\":\"opencv\",\"index_or_path\":${LEFT_IDX},\"width\":640,\"height\":480,\"fps\":30,\"rotation\":0},\"middle\":{\"type\":\"opencv\",\"index_or_path\":${MIDDLE_IDX},\"width\":640,\"height\":480,\"fps\":30,\"rotation\":0},\"right\":{\"type\":\"opencv\",\"index_or_path\":${RIGHT_IDX},\"width\":640,\"height\":480,\"fps\":30,\"rotation\":0}}" \
    --dataset.repo_id="$DATASET_ID" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=0 \
    --dataset.fps=30 \
    --dataset.video=true \
    --dataset.single_task="Place the cube to the other side." \
    --dataset.push_to_hub=false \
    --policy.type=act \
    --policy.device=cuda \
    --policy.chunk_size=100 \
    --policy.n_action_steps=75 \
    --policy.n_obs_steps=1 \
    --policy.pretrained_path="$CHECKPOINT" \
    --display_data=false \
    --show_control_window=false \
    --play_sounds=false

echo
echo "Inference and connected right-arm reset completed successfully."