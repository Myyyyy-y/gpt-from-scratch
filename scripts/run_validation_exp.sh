#!/bin/bash
# Launcher for training-technique validation experiments (29M base, aligned
# with champion config 003).
#
# Usage (recommended inside tmux):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_validation_exp.sh combo
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_validation_exp.sh zeroinit
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_validation_exp.sh untie
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_validation_exp.sh qknorm_clean
#   SEED=42 CUDA_VISIBLE_DEVICES=4 bash scripts/run_validation_exp.sh combo   # custom seed
#
# Experiments:
#   combo        combo: QK-norm + Muon + ReLU² + untie + zero-init stacked
#   zeroinit     zero-init output projection (vs 003 champion)
#   untie        untied embedding/lm_head (vs 003 champion)
#   qknorm_clean QK-norm clean attribution: champion config (lr 1e-3, min_lr 3e-5) + --qk_norm
#   layernorm    norm ablation: RMSNorm -> LayerNorm
#   data3m       data-size ablation: only the first 3M training tokens
#   champion     champion rerun (SEED=1/2/3 for significance testing)
#   attnres      Attention Residuals deep residual routing training
#   kvbench      29M GPU KV-cache re-test (evaluation only, no training)

set -euo pipefail

NAME="${1:?usage: bash scripts/run_validation_exp.sh <combo|zeroinit|untie|qknorm_clean|layernorm|data3m|champion|attnres|kvbench>}"
SEED="${SEED:-0}"

# 29M champion base (003): 8 layers / d_model 512 / 8 heads / d_ff 1344, lr 1e-3
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
        echo ">>> 29M GPU KV-cache re-test (bench_tokens=256)"
        "${PYTHON:-python}" -m src.eval --ckpt experiments/003_29m_lr_1e3/best.pt \
            --data_dir data --bench_tokens 256 --device cuda
        exit 0
        ;;
    *)
        echo "unknown experiment: ${NAME}" >&2
        exit 1
        ;;
esac

echo ">>> ${NAME} -> ${OUT} (seed=${SEED})"
"${PYTHON:-python}" -m src.train --out_dir "${OUT}" "${BASE[@]}" "${EXTRA[@]}"
