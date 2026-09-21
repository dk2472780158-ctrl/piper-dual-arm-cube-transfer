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

SOURCE_DATASET="${LEROBOT_HOME}/local/cube_r2l_dataset"
ROBUST_DATASET="${LEROBOT_HOME}/local/cube_r2l_dataset_robust_v1"
ROBUST_REPO_ID="local/cube_r2l_dataset_robust_v1"

BASE_CHECKPOINT="$PROJECT_ROOT/outputs/train/cube_r2l_act_v2_robust_c100/checkpoints/040000/pretrained_model"

OUTPUT_DIR="$PROJECT_ROOT/outputs/train/cube_r2l_generalized_ft20k"
LOG_FILE="$PROJECT_ROOT/cube_r2l_generalized_ft20k.log"

cd "$PROJECT_ROOT"

if [[ ! -f "$SOURCE_DATASET/meta/stats.json" ]]; then
    echo "ERROR: source dataset is invalid:"
    echo "$SOURCE_DATASET"
    exit 1
fi

if [[ ! -f "$BASE_CHECKPOINT/config.json" ]]; then
    echo "ERROR: base checkpoint not found:"
    echo "$BASE_CHECKPOINT"
    exit 1
fi

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "ERROR: output directory already exists:"
    echo "$OUTPUT_DIR"
    echo "Please rename OUTPUT_DIR before starting another run."
    exit 1
fi

if [[ ! -e "$ROBUST_DATASET" ]]; then
    echo "Copying dataset..."
    cp -a --reflink=auto "$SOURCE_DATASET" "$ROBUST_DATASET"
else
    echo "Using existing robust dataset:"
    echo "$ROBUST_DATASET"
fi

export ROBUST_DATASET

python - <<'PY'
import json
import os
import shutil
from pathlib import Path

import numpy as np

dataset_root = Path(os.environ["ROBUST_DATASET"])
stats_path = dataset_root / "meta" / "stats.json"
backup_path = dataset_root / "meta" / "stats.original.json"

if not backup_path.exists():
    shutil.copy2(stats_path, backup_path)

with stats_path.open("r", encoding="utf-8") as file:
    stats = json.load(file)

feature = stats.get("observation.state")
if feature is None:
    raise KeyError("stats.json does not contain observation.state")

original_std = np.asarray(feature["std"], dtype=np.float64)
original_shape = original_std.shape
std = original_std.reshape(-1)

if std.size != 28:
    raise ValueError(
        f"Expected observation.state std to contain 28 values, got {std.size}"
    )

# observation.state:
# 0,2,4...26 = joint/gripper positions
# 1,3,5...27 = effort values
position_indices = np.arange(0, 28, 2)
effort_indices = np.arange(1, 28, 2)

position_floor = 0.05
effort_floor = 0.10

before = std.copy()
std[position_indices] = np.maximum(std[position_indices], position_floor)
std[effort_indices] = np.maximum(std[effort_indices], effort_floor)

feature["std"] = std.reshape(original_shape).tolist()

with stats_path.open("w", encoding="utf-8") as file:
    json.dump(stats, file, ensure_ascii=False, indent=4)
    file.write("\n")

print("\n===== Robust normalization =====")
print(f"stats: {stats_path}")
print(f"backup: {backup_path}")
print(f"position std floor: {position_floor}")
print(f"effort std floor:   {effort_floor}")
print("\nChanged dimensions:")

for index, (old, new) in enumerate(zip(before, std, strict=True)):
    if new != old:
        kind = "position" if index % 2 == 0 else "effort"
        print(f"  index={index:2d} type={kind:8s} {old:.8f} -> {new:.8f}")
PY

echo
echo "Checking dataset..."

python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local/cube_r2l_dataset_robust_v1")

print("episodes:", dataset.meta.total_episodes)
print("frames:", len(dataset))
print("tasks:")
print(dataset.meta.tasks)
PY

echo
echo "Starting 10000-step fine-tuning..."
echo "Base checkpoint: $BASE_CHECKPOINT"
echo "Dataset:        $ROBUST_REPO_ID"
echo "Output:         $OUTPUT_DIR"
echo "Log:            $LOG_FILE"
echo

lerobot-train \
    --dataset.repo_id="$ROBUST_REPO_ID" \
    --policy.path="$BASE_CHECKPOINT" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.chunk_size=100 \
    --policy.n_action_steps=50 \
    --policy.n_obs_steps=1 \
    --output_dir="$OUTPUT_DIR" \
    --job_name=cube_r2l_generalized_ft20k \
    --batch_size=8 \
    --steps=20000 \
    --optimizer.lr=3e-6 \
    --save_checkpoint=true \
    --save_freq=5000 \
    --eval_freq=0 \
    --wandb.enable=false \
    2>&1 | tee "$LOG_FILE"
