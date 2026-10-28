#!/usr/bin/env bash
# 在「真机 50 集 + 仿真 80 集」的合并集上微调 ACT，做 sim2real 混合训练。
#
# 和 train_sim_act.sh 的唯一区别是数据集换成了 cube_mix_v1，其余超参与
# 仿真那轮逐项相同（同一起点 checkpoint、同 lr、同 batch、同步数、同 seed），
# 这样两边结果可以直接对比：只换了数据。
#
#   bash train_mix_act.sh              # 12k 步
set -euo pipefail
ROOT=/home/databall/VLA/lerobot_piper
ENV=/home/databall/miniconda3/envs/lerobot_v30
cd "$ROOT"
export PATH="$ENV/bin:$PATH"
export PYTHONUNBUFFERED=1

DS=/home/databall/.cache/huggingface/lerobot/local/cube_mix_v1
FROM_REAL=/home/databall/VLA/lerobot_piper/outputs/train/cube_r2l_new_env_ft_from_80ep20k/checkpoints/020000/pretrained_model

NAME="${NAME:-cube_mix_act_v1_ft}"
STEPS="${STEPS:-12000}"

[[ -d "$DS" ]] || { echo "ERROR: 找不到数据集 $DS" >&2; exit 1; }
[[ -d "$FROM_REAL" ]] || { echo "ERROR: 找不到起点 checkpoint $FROM_REAL" >&2; exit 1; }
if [[ -e outputs/train/$NAME ]]; then
    echo "ERROR: outputs/train/$NAME already exists" >&2
    exit 1
fi

echo "数据集 $DS"
echo "起点   $FROM_REAL"
echo "输出   outputs/train/$NAME  步数 $STEPS"

lerobot-train \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/cube_mix_v1 \
  --dataset.root="$DS" \
  --policy.type=act \
  --output_dir=outputs/train/$NAME \
  --job_name=$NAME \
  --policy.device=cuda \
  --seed=1000 \
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
  --policy.pretrained_path="$FROM_REAL" \
  --save_checkpoint=true \
  --save_freq=4000 \
  --wandb.enable=false \
  --eval_freq=0 \
  --num_workers=4
