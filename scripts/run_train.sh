#!/bin/bash
# 正式训练启动脚本（模板）
#
# 用法：
#   bash scripts/run_train.sh                      # 默认 baseline 配置
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_train.sh   # 指定显卡
#
# 建议在 tmux 里跑（断连不中断）：
#   tmux new -s train
#   bash scripts/run_train.sh
#   （按 Ctrl+B 再按 D 脱离；tmux attach -t train 重新连接）

set -euo pipefail   # 任何一步出错立即退出；未定义变量视为错误

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
