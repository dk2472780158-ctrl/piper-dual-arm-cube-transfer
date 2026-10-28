#!/usr/bin/env bash
# Train ACT on the episodes generated in the simulated cell.
#
# Fine-tunes from the real-cell checkpoint rather than starting cold: the task
# and the action layout are the same, only the pictures differ, so the weights
# are a better starting point than noise and 12k steps is enough to adapt.
#
#   bash train_sim_act.sh                     # fine-tune, 12k steps
#   bash train_sim_act.sh --from-scratch      # cold start, 20k steps
set -euo pipefail
ROOT=/home/databall/VLA/lerobot_piper
ENV=/home/databall/miniconda3/envs/lerobot_v30
cd "$ROOT"
export PATH="$ENV/bin:$PATH"
export PYTHONUNBUFFERED=1

DS="${DS:-/home/databall/.cache/huggingface/lerobot/local/cube_sim_v1}"
FROM_REAL=/home/databall/VLA/lerobot_piper/outputs/train/cube_r2l_new_env_ft_from_80ep20k/checkpoints/020000/pretrained_model

if [[ "${1:-}" == "--from-scratch" ]]; then
    NAME="${NAME:-cube_sim_act_v1_scratch}"
    STEPS="${STEPS:-20000}"
    EXTRA=()
else
    NAME="${NAME:-cube_sim_act_v1_ft}"
    STEPS="${STEPS:-12000}"
    EXTRA=(--policy.pretrained_path="$FROM_REAL")
fi

if [[ -e outputs/train/$NAME ]]; then
    echo "ERROR: outputs/train/$NAME already exists" >&2
    exit 1
fi

lerobot-train \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/$(basename "$DS") \
  --dataset.root="$DS" \
  --policy.type=act \
  --output_dir=outputs/train/$NAME \
  --job_name=$NAME \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=$STEPS \
  --optimizer.lr=1e-5 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=50 \
  --policy.n_obs_steps=1 \
  --policy.dim_model=512 \
  --policy.n_heads=8 \
  --policy.dim_feedforward=3200 \
  --policy.vision_backbone=resnet18 \
  "${EXTRA[@]}" \
  --save_checkpoint=true \
  --save_freq=4000 \
  --wandb.enable=false \
  --eval_freq=0 \
  --num_workers=4
