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

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"

# 停止空闲相机服务，推理时由当前进程向WebRTC发布画面
# shellcheck source=../ops/fanout/scripts/piper_lerobot_fanout_env.sh
source "${PIPER_FANOUT_ROOT}/scripts/piper_lerobot_fanout_env.sh"

CHECKPOINT="$PROJECT_ROOT/outputs/train/cube_l2r_left_arm_act_v1_robust_c100/checkpoints/050000/pretrained_model"

if [[ ! -f "$CHECKPOINT/config.json" ]]; then
    echo "ERROR: checkpoint不存在："
    echo "$CHECKPOINT"
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/src/lerobot/scripts/piper_cube_record.py" ]]; then
    echo "ERROR: 缺少专用推理程序："
    echo "$PROJECT_ROOT/src/lerobot/scripts/piper_cube_record.py"
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/reset_piper_pose.py" ]]; then
    echo "ERROR: 缺少归位程序："
    echo "$PROJECT_ROOT/reset_piper_pose.py"
    exit 1
fi

# 1表示不增加额外动作低通滤波。
# 平滑交给ACT Temporal Ensembling完成。
export PIPER_ACTION_FILTER_ALPHA="${PIPER_ACTION_FILTER_ALPHA:-1}"

# 新checkpoint已经使用稳健归一化，关闭临时state裁剪。
export PIPER_CLIP_OBSERVATION_STATE="${PIPER_CLIP_OBSERVATION_STATE:-0}"

resolve_piper_cam() {
    local camera_path="$1"
    local resolved
    local device_id
    local video_device

    if [[ ! -e "$camera_path" ]]; then
        echo "ERROR: 相机不存在：$camera_path" >&2
        return 1
    fi

    if [[ -L "$camera_path" ]]; then
        resolved="$(readlink -f "$camera_path")"

        if [[ ! "$resolved" =~ ^/dev/video[0-9]+$ ]]; then
            echo "ERROR: $camera_path -> $resolved，不是有效video设备" >&2
            return 1
        fi

        if [[ ! -e "$resolved" ]]; then
            echo "ERROR: 相机目标不存在：$resolved" >&2
            return 1
        fi

        echo "$resolved"
        return 0
    fi

    # 兼容平台将/dev/camera_*以字符设备方式挂载，而不是软链接。
    device_id="$(stat -c '%t:%T' "$camera_path")"

    for video_device in /dev/video*; do
        [[ -e "$video_device" ]] || continue

        if [[ "$(stat -c '%t:%T' "$video_device")" == "$device_id" ]]; then
            echo "$video_device"
            return 0
        fi
    done

    echo "ERROR: 无法将$camera_path映射为/dev/videoN" >&2
    return 1
}

LEFT_DEV="$(resolve_piper_cam /dev/camera_left)"
MIDDLE_DEV="$(resolve_piper_cam /dev/camera_middle)"
RIGHT_DEV="$(resolve_piper_cam /dev/camera_right)"

LEFT_IDX="${LEFT_DEV#/dev/video}"
MIDDLE_IDX="${MIDDLE_DEV#/dev/video}"
RIGHT_IDX="${RIGHT_DEV#/dev/video}"

RUN_NAME="eval_cube_l2r_left_050000_$(date +%Y%m%d_%H%M%S)"
DATASET_ID="local/${RUN_NAME}"

echo "left=$LEFT_DEV"
echo "middle=$MIDDLE_DEV"
echo "right=$RIGHT_DEV"
echo "checkpoint=$CHECKPOINT"
echo "result=$DATASET_ID"
echo "filter alpha=$PIPER_ACTION_FILTER_ALPHA"
echo "state clipping=$PIPER_CLIP_OBSERVATION_STATE"
echo "temporal ensemble coefficient=0.01"
echo "automatic reset arm=left"
echo
echo "WARNING: 请清空工作区域，并保证急停按钮随时可用。"
echo

python -m lerobot.scripts.piper_cube_record \
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
    --display_data=false \
    --show_control_window=false \
    --play_sounds=false \
    --auto_reset_arm=left \
    --reset_joint_speed=0.80 \
    --reset_gripper_speed=0.01 \
    --reset_fps=30 \
    --reset_settle_timeout=8 \
    --reset_settle_tolerance=0.02 \
    --reset_gripper_tracking_limit=0.01

echo
echo "左臂推理及自动归位全部完成。"
