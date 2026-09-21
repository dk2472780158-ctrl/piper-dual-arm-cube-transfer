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

# 原来训练50条数据使用的数据集
OLD_DATASET="local/cube_r2l_dataset_v2_robust"

# 新增的3组数据，每组约10条；名字虽然写成l2r，但实际是r2l
NEW_DATASET_5="local/cube_l2r_5"
NEW_DATASET_6="local/cube_l2r_6"
NEW_DATASET_7="local/cube_l2r_7"

# 合并后的80条数据
MERGED_REPO_ID="local/cube_r2l_dataset_v3_80ep"
MERGED_ROOT="$LEROBOT_HOME/$MERGED_REPO_ID"

# 原来50000步模型
BASE_CHECKPOINT="$PROJECT_ROOT/outputs/train/cube_r2l_act_v2_robust_c100/checkpoints/050000/pretrained_model"

# 默认微调20000步
STEPS="${FINETUNE_STEPS:-20000}"

OUTPUT_DIR="$PROJECT_ROOT/outputs/train/cube_r2l_act_v3_80ep_ft${STEPS}"
JOB_NAME="cube_r2l_act_v3_80ep_ft${STEPS}"

cd "$PROJECT_ROOT"
export HF_LEROBOT_HOME="$LEROBOT_HOME"

if [[ ! -f "$BASE_CHECKPOINT/config.json" ]]; then
    echo "ERROR: 找不到50000步checkpoint："
    echo "$BASE_CHECKPOINT"
    exit 1
fi

echo "===== 检查四组输入数据 ====="

python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_ids = [
    "local/cube_r2l_dataset_v2_robust",
    "local/cube_l2r_5",
    "local/cube_l2r_6",
    "local/cube_l2r_7",
]

reference = None
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

    if reference is None:
        reference = {
            "fps": dataset.meta.fps,
            "features": features,
        }
    else:
        if dataset.meta.fps != reference["fps"]:
            raise RuntimeError(
                f"{repo_id} FPS不一致："
                f"{dataset.meta.fps} != {reference['fps']}"
            )

        if features != reference["features"]:
            raise RuntimeError(
                f"{repo_id} 的特征、维度或关节顺序与旧数据不一致"
            )

    total_episodes += dataset.meta.total_episodes
    total_frames += len(dataset)

print("\n===== 检查通过 =====")
print("总episodes:", total_episodes)
print("总frames:", total_frames)

if total_episodes != 80:
    print(
        f"WARNING: 总数不是预期的80条，而是{total_episodes}条。"
        "请确认是否有缺失或重复数据。"
    )
PY

echo
echo "===== 合并数据集 ====="

if [[ ! -f "$MERGED_ROOT/meta/info.json" ]]; then
    lerobot-edit-dataset \
        --repo_id="$MERGED_REPO_ID" \
        --operation.type=merge \
        --operation.repo_ids="['$OLD_DATASET','$NEW_DATASET_5','$NEW_DATASET_6','$NEW_DATASET_7']" \
        --push_to_hub=false
else
    echo "合并数据集已经存在，跳过合并："
    echo "$MERGED_ROOT"
fi

echo
echo "===== 修正左臂归一化 ====="

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
    raise FileNotFoundError(stats_path)

# 第一次执行时保存原始统计量
if not backup_path.exists():
    shutil.copy2(stats_path, backup_path)

with stats_path.open("r", encoding="utf-8") as file:
    stats = json.load(file)

feature = stats.get("observation.state")
if feature is None:
    raise KeyError("stats.json中没有observation.state")

original = np.asarray(feature["std"], dtype=np.float64)
shape = original.shape
std = original.reshape(-1)

if std.size != 28:
    raise ValueError(
        f"observation.state应该有28维，实际为{std.size}维"
    )

# 前14维是左臂：
# 偶数索引是position，奇数索引是effort
left_position_indices = np.arange(0, 14, 2)
left_effort_indices = np.arange(1, 14, 2)

# 防止静止左臂的微小反馈误差被归一化放大数百/数千倍
position_std_floor = 0.05
effort_std_floor = 0.10

before = std.copy()

std[left_position_indices] = np.maximum(
    std[left_position_indices],
    position_std_floor,
)

std[left_effort_indices] = np.maximum(
    std[left_effort_indices],
    effort_std_floor,
)

feature["std"] = std.reshape(shape).tolist()

with stats_path.open("w", encoding="utf-8") as file:
    json.dump(stats, file, ensure_ascii=False, indent=4)
    file.write("\n")

print("stats:", stats_path)
print("backup:", backup_path)
print("\n被修正的左臂维度：")

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
    print("  不需要修改，现有标准差已经足够大。")
PY

echo
echo "===== 验证最终数据集 ====="

python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local/cube_r2l_dataset_v3_80ep")

print("episodes:", dataset.meta.total_episodes)
print("frames:", len(dataset))
print("fps:", dataset.meta.fps)
print("features:", list(dataset.features))
print("tasks:")
print(dataset.meta.tasks)

if dataset.meta.total_episodes != 80:
    print("WARNING: episode总数不是80，请确认后再判断训练结果。")
PY

if [[ -e "$OUTPUT_DIR" ]]; then
    echo
    echo "ERROR: 训练输出目录已经存在："
    echo "$OUTPUT_DIR"
    echo "为防止覆盖已有checkpoint，本次停止。"
    exit 1
fi

echo
echo "===== 开始微调 ====="
echo "checkpoint=$BASE_CHECKPOINT"
echo "dataset=$MERGED_REPO_ID"
echo "steps=$STEPS"
echo "output=$OUTPUT_DIR"
echo

lerobot-train \
    --dataset.repo_id="$MERGED_REPO_ID" \
    --policy.path="$BASE_CHECKPOINT" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.chunk_size=100 \
    --policy.n_action_steps=50 \
    --policy.n_obs_steps=1 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --batch_size=8 \
    --steps="$STEPS" \
    --optimizer.lr=3e-6 \
    --save_checkpoint=true \
    --save_freq=10000 \
    --eval_freq=0 \
    --wandb.enable=false
