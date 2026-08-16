#!/bin/bash
# P1 训练技术验证实验启动脚本（29M 基座，全部对齐冠军组 003 配置）
#
# 用法（建议在 tmux 里跑）：
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_validation_exp.sh combo
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_validation_exp.sh zeroinit
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_validation_exp.sh untie
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_validation_exp.sh qknorm_clean
#   SEED=42 CUDA_VISIBLE_DEVICES=4 bash scripts/run_validation_exp.sh combo   # 改 seed
#
# 可跑项：
#   combo        技术组合实验：QK-norm + Muon + ReLU² + untie + zero-init 全叠加
#   zeroinit     zero-init 单项消融（vs 003 冠军）
#   untie        untie 单项消融（vs 003 冠军）
#   qknorm_clean QK-norm 干净归因：冠军配置（lr 1e-3, min_lr 3e-5）+ --qk_norm
#   layernorm    归一化消融：RMSNorm -> LayerNorm
#   data3m       数据量消融：只用训练集前 300 万 token
#   champion     冠军组复跑（配合 SEED=1/2/3 做显著性检验）
#   attnres      Attention Residuals 深度残差路由训练
#   kvbench      29M GPU 版 KV cache 复测（评估型，非训练）

set -euo pipefail

NAME="${1:?用法: bash scripts/run_validation_exp.sh <combo|zeroinit|untie|qknorm_clean|layernorm|data3m|champion|attnres|kvbench>}"
SEED="${SEED:-0}"

# 29M 冠军基座（003）：8 层 / d_model 512 / 8 头 / d_ff 1344，lr 1e-3
BASE=(
    --data_dir data
    --n_layers 8 --d_model 512 --n_heads 8 --d_ff 1344
    --max_steps 28500 --batch_size 64 --context_length 256
    --max_lr 1e-3 --min_lr 3e-5 --warmup_steps 200
    --weight_decay 0.1 --grad_clip 1.0
    --eval_interval 250 --save_interval 1000
    --dtype bf16 --seed "${SEED}"
)

case "${NAME}" in
    combo)
        OUT=experiments/012_combo
        EXTRA=(--qk_norm --ffn_type relu2 --optimizer muon --muon_lr 0.01 --untie --zero_init_proj)
        ;;
    zeroinit)
        OUT=experiments/012_zeroinit
        EXTRA=(--zero_init_proj)
        ;;
    untie)
        OUT=experiments/012_untied
        EXTRA=(--untie)
        ;;
    qknorm_clean)
        OUT=experiments/012_qknorm_clean
        EXTRA=(--qk_norm)
        ;;
    layernorm)
        OUT=experiments/013_layernorm
        EXTRA=(--norm_type layernorm)
        ;;
    data3m)
        OUT=experiments/013_data_3m
        EXTRA=(--train_limit 3000000)
        ;;
    champion)
        OUT=experiments/013_champion_seed${SEED}
        EXTRA=()
        ;;
    attnres)
        OUT=experiments/013_attnres
        EXTRA=(--attn_res)
        ;;
    kvbench)
        echo ">>> 29M GPU 版 KV cache 复测（bench_tokens=256）"
        "${PYTHON:-python}" -m src.eval --ckpt experiments/003_29m_lr_1e3/best.pt \
            --data_dir data --bench_tokens 256 --device cuda
        exit 0
        ;;
    *)
        echo "未知实验名: ${NAME}" >&2
        exit 1
        ;;
esac

echo ">>> ${NAME} -> ${OUT} (seed=${SEED})"
"${PYTHON:-python}" -m src.train --out_dir "${OUT}" "${BASE[@]}" "${EXTRA[@]}"
