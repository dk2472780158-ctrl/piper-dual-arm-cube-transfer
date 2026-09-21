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

# Pin the existing control environment; no conda activation is required.
ENV_ROOT="${LEROBOT_ENV}"
export PATH="$ENV_ROOT/bin:$PATH"
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
    echo "ERROR: control Python not found: $ENV_ROOT/bin/python" >&2
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"

# 停止空闲相机服务，并允许推理期间发布 WebRTC 画面
# shellcheck source=../ops/fanout/scripts/piper_lerobot_fanout_env.sh
source "${PIPER_FANOUT_ROOT}/scripts/piper_lerobot_fanout_env.sh"

CHECKPOINT="$PROJECT_ROOT/outputs/train/cube_r2l_new_env_ft_from_80ep20k/checkpoints/020000/pretrained_model"

INFERENCE_MODULE="$PROJECT_ROOT/src/lerobot/scripts/piper_cube_record.py"
RESET_MODULE="$PROJECT_ROOT/reset_piper_pose.py"

if [[ ! -s "$CHECKPOINT/config.json" || ! -s "$CHECKPOINT/model.safetensors" ]]; then
    echo "ERROR: checkpoint not found:"
    echo "$CHECKPOINT"
    exit 1
fi

if [[ ! -f "$INFERENCE_MODULE" ]]; then
    echo "ERROR: dedicated inference module not found:"
    echo "$INFERENCE_MODULE"
    exit 1
fi

if [[ ! -f "$RESET_MODULE" ]]; then
    echo "ERROR: reset module not found:"
    echo "$RESET_MODULE"
    exit 1
fi

# Temporal Ensembling 已负责平滑，因此关闭额外动作滤波
export PIPER_ACTION_FILTER_ALPHA=1

# 使用新 checkpoint 保存的稳健归一化，不启用临时裁剪
export PIPER_CLIP_OBSERVATION_STATE=0

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

RUN_NAME="eval_cube_r2l_new_env_020000_$(date +%Y%m%d_%H%M%S)_$$"
DATASET_ID="local/${RUN_NAME}"

echo "=================================================="
echo "Piper ACT temporal-ensemble inference"
echo "=================================================="
echo "left=$LEFT_DEV"
echo "middle=$MIDDLE_DEV"
echo "right=$RIGHT_DEV"
echo "fanout ENABLE_PIPER_RGB_PUBLISH=${ENABLE_PIPER_RGB_PUBLISH:-unset}"
echo "checkpoint=$CHECKPOINT"
echo "result=$DATASET_ID"
echo "filter alpha=$PIPER_ACTION_FILTER_ALPHA"
echo "state clipping=$PIPER_CLIP_OBSERVATION_STATE"
echo "chunk size=100"
echo "action steps=1"
echo "temporal ensemble coefficient=0.01"
echo "automatic reset arm=right"
echo
echo "WARNING: Clear the workspace and keep the emergency stop ready."
echo

# --- [新增] Rerun 实时监控 ---
# 1) 显式 pin 控制解释器到宿主机 ext4 那份（有 matplotlib 3.11.1）
# 2) source rerun_env.sh 把 SDK 顶到 0.31.3 并设 PYTHONPYCACHEPREFIX
# 原脚本没有这一段；不 source 的话 display_data=true 会因 0.22.1 无 rr.Scalars 而崩。
export PATH="${LEROBOT_ENV}/bin:$PATH"
source "${CUBE_ENVFIX_ROOT}/rerun_env.sh"

"$ENV_ROOT/bin/python" -m lerobot.scripts.piper_cube_record \
    --auto_reset_arm=right \
    --reset_joint_speed=0.8 \
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
    --policy.n_action_steps=1 \
    --policy.n_obs_steps=1 \
    --policy.temporal_ensemble_coeff=0.01 \
    --policy.pretrained_path="$CHECKPOINT" \
    --display_data=true \
    --display_ip=127.0.0.1 \
    --display_port=9876 \
    --show_control_window=false \
    --play_sounds=false

echo
echo "Temporal-ensemble inference and right-arm reset completed successfully."
