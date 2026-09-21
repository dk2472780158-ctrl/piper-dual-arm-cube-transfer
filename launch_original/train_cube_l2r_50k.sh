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
LEROBOT_HOME="${LEROBOT_HOME}"

INPUT_0="local/cube_l2r_0"
INPUT_1="local/cube_l2r_1"
INPUT_2="local/cube_l2r_2"
INPUT_3="local/cube_l2r_3"
INPUT_4="local/cube_l2r_4"

# 名字明确标注：左臂执行从左到右
MERGED_REPO_ID="local/cube_l2r_left_arm_dataset_v1_robust"
MERGED_ROOT="$LEROBOT_HOME/$MERGED_REPO_ID"

OUTPUT_DIR="$PROJECT_ROOT/outputs/train/cube_l2r_left_arm_act_v1_robust_c100"
JOB_NAME="cube_l2r_left_arm_act_v1_robust_c100"

cd "$PROJECT_ROOT"

export HF_LEROBOT_HOME="$LEROBOT_HOME"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "===== 检查5组原始数据 ====="

python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_ids = [
    "local/cube_l2r_0",
    "local/cube_l2r_1",
    "local/cube_l2r_2",
    "local/cube_l2r_3",
    "local/cube_l2r_4",
]

reference_fps = None
reference_features = None
total_episodes = 0
total_frames = 0

for repo_id in repo_ids:
    dataset = LeRobotDataset(repo_id)

    features = {
        name: (
            tuple(feature.get("shape", [])),
            feature.get("dtype"),
            tuple(feature.get("names") or []),
        )
        for name, feature in dataset.features.items()
    }

    print(f"\n{repo_id}")
    print("  episodes:", dataset.meta.total_episodes)
    print("  frames:", len(dataset))
    print("  fps:", dataset.meta.fps)
    print("  tasks:")
    print(dataset.meta.tasks)

    if reference_fps is None:
        reference_fps = dataset.meta.fps
        reference_features = features
    else:
        if dataset.meta.fps != reference_fps:
            raise RuntimeError(
                f"{repo_id} 的FPS不一致："
                f"{dataset.meta.fps} != {reference_fps}"
            )

        if features != reference_features:
            raise RuntimeError(
                f"{repo_id} 的特征、维度或关节顺序不一致"
            )

    total_episodes += dataset.meta.total_episodes
    total_frames += len(dataset)

print("\n===== 原始数据检查通过 =====")
print("总episodes:", total_episodes)
print("总frames:", total_frames)

if total_episodes != 50:
    print(
        f"WARNING: 预期为50条，实际为{total_episodes}条。"
        "请确认是否存在缺失或重复数据。"
    )
PY

echo
echo "===== 合并数据 ====="

if [[ -e "$MERGED_ROOT" ]]; then
    echo "ERROR: 合并目标已经存在："
    echo "$MERGED_ROOT"
    echo
    echo "为防止误用旧数据，本次停止。"
    echo "确认它是本次正确数据后，可以修改脚本中的MERGED_REPO_ID。"
    exit 1
fi

lerobot-edit-dataset \
    --repo_id="$MERGED_REPO_ID" \
    --operation.type=merge \
    --operation.repo_ids="['$INPUT_0','$INPUT_1','$INPUT_2','$INPUT_3','$INPUT_4']" \
    --push_to_hub=false

echo
echo "===== 修正右臂归一化 ====="

export MERGED_ROOT

python - <<'PY'
import json
import os
import shutil
from pathlib import Path

import numpy as np

root = Path(os.environ["MERGED_ROOT"])
stats_path = root / "meta" / "stats.json"
backup_path = root / "meta" / "stats.before_robust_normalization.json"

if not stats_path.exists():
    raise FileNotFoundError(f"找不到统计文件：{stats_path}")

shutil.copy2(stats_path, backup_path)

with stats_path.open("r", encoding="utf-8") as file:
    stats = json.load(file)

feature = stats.get("observation.state")
if feature is None:
    raise KeyError("stats.json中没有observation.state")

original_std = np.asarray(feature["std"], dtype=np.float64)
original_shape = original_std.shape
std = original_std.reshape(-1)

if std.size != 28:
    raise ValueError(
        f"observation.state应该是28维，实际是{std.size}维"
    )

# observation.state布局：
#
# 0～13：左臂
#   0,2,4...12  = 左臂position
#   1,3,5...13  = 左臂effort
#
# 14～27：右臂
#   14,16...26  = 右臂position
#   15,17...27  = 右臂effort
#
# 本任务中左臂运动、右臂基本静止，
# 所以只修正右臂输入状态的标准差。

right_position_indices = np.arange(14, 28, 2)
right_effort_indices = np.arange(15, 28, 2)

position_std_floor = 0.05
effort_std_floor = 0.10

before = std.copy()

std[right_position_indices] = np.maximum(
    std[right_position_indices],
    position_std_floor,
)

std[right_effort_indices] = np.maximum(
    std[right_effort_indices],
    effort_std_floor,
)

feature["std"] = std.reshape(original_shape).tolist()

with stats_path.open("w", encoding="utf-8") as file:
    json.dump(stats, file, ensure_ascii=False, indent=4)
    file.write("\n")

print("统计文件:", stats_path)
print("原始备份:", backup_path)
print("右臂position标准差下限:", position_std_floor)
print("右臂effort标准差下限:", effort_std_floor)

print("\n被修正的右臂维度：")

changed = 0

for index, (old, new) in enumerate(zip(before, std, strict=True)):
    if old != new:
        kind = "position" if index % 2 == 0 else "effort"

        print(
            f"  index={index:2d} "
            f"type={kind:8s} "
            f"{old:.8f} -> {new:.8f}"
        )

        changed += 1

if changed == 0:
    print("  当前右臂标准差已经足够大，无须调整。")
PY

echo
echo "===== 检查合并后的动作方向 ====="

export MERGED_ROOT

python - <<'PY'
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path(os.environ["MERGED_ROOT"])
files = sorted((root / "data").glob("**/*.parquet"))

if not files:
    raise FileNotFoundError("合并数据集中没有parquet文件")

table = pq.read_table(files, columns=["action", "episode_index"])
action = np.asarray(table["action"].to_pylist(), dtype=np.float64)

if action.ndim != 2 or action.shape[1] != 14:
    raise ValueError(f"action应该是(N,14)，实际为{action.shape}")

if not np.isfinite(action).all():
    raise ValueError("action中存在NaN或Inf")

names = [
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper",
]

ranges = np.ptp(action, axis=0)

print("frames:", len(action))
print("\naction范围：")

for name, value in zip(names, ranges, strict=True):
    print(f"  {name:16s} {value:.6f}")

left_range = float(np.max(ranges[:7]))
right_range = float(np.max(ranges[7:]))

print()
print("左臂最大动作范围:", left_range)
print("右臂最大动作范围:", right_range)

if left_range <= right_range:
    print()
    print("WARNING: 数据中右臂动作范围不小于左臂。")
    print("请确认这批数据确实是左臂执行任务后再训练。")
else:
    print("PASS: 数据主要由左臂运动。")
PY

echo
echo "===== 最终数据集信息 ====="

python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "local/cube_l2r_left_arm_dataset_v1_robust"
)

print("episodes:", dataset.meta.total_episodes)
print("frames:", len(dataset))
print("fps:", dataset.meta.fps)
print("features:", list(dataset.features))
print("tasks:")
print(dataset.meta.tasks)
PY

if [[ -e "$OUTPUT_DIR" ]]; then
    echo
    echo "ERROR: 训练目录已经存在："
    echo "$OUTPUT_DIR"
    echo "为防止覆盖checkpoint，本次停止。"
    exit 1
fi

echo
echo "===== 开始从头训练ACT ====="
echo "dataset=$MERGED_REPO_ID"
echo "output=$OUTPUT_DIR"
echo "steps=50000"
echo

lerobot-train \
    --dataset.repo_id="$MERGED_REPO_ID" \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.chunk_size=100 \
    --policy.n_action_steps=50 \
    --policy.n_obs_steps=1 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --batch_size=8 \
    --steps=50000 \
    --optimizer.lr=1e-5 \
    --save_checkpoint=true \
    --save_freq=10000 \
    --eval_freq=0 \
    --wandb.enable=false
