#!/bin/bash
# Template for launching a real training run (recommend running inside tmux).
set -euo pipefail

python -m src.train \
    --data_dir data \
    --out_dir experiments/001_baseline \
    --max_steps 28500 \
    --batch_size 64 \
    --max_lr 3e-4 \
    --min_lr 3e-5 \
    --warmup_steps 200 \
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    --eval_interval 250 \
    --save_interval 1000 \
    --dtype bf16
