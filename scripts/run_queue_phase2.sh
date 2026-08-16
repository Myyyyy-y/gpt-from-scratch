#!/bin/bash
# P2/P3 GPU 队列：等待 GPU 空闲后，按序跑完剩余 GPU 任务（约 10 小时）
#
# 顺序：layernorm 消融 -> 数据量消融 -> AttnRes 训练 -> 冠军组 3 seed
#       -> 29M GPU 版 KV cache 复测
#
# 用法：
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_queue_phase2.sh

set -euo pipefail

GPU_ID="${CUDA_VISIBLE_DEVICES:?请设置 CUDA_VISIBLE_DEVICES}"

wait_gpu_free() {
    echo ">>> [$(date '+%F %T')] 等待 GPU ${GPU_ID} 空闲 ..."
    while :; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                -i "${GPU_ID}" 2>/dev/null | head -1)
        if [ -n "${used}" ] && [ "${used}" -lt 500 ]; then
            echo ">>> [$(date '+%F %T')] GPU ${GPU_ID} 空闲（${used} MiB），开始队列"
            return
        fi
        sleep 120
    done
}

wait_gpu_free

for NAME in layernorm data3m attnres; do
    echo "########## [$(date '+%F %T')] 开始 ${NAME} ##########"
    bash scripts/run_validation_exp.sh "${NAME}"
    echo "########## [$(date '+%F %T')] 完成 ${NAME} ##########"
done

for SEED in 1 2 3; do
    echo "########## [$(date '+%F %T')] 开始 champion seed=${SEED} ##########"
    SEED="${SEED}" bash scripts/run_validation_exp.sh champion
    echo "########## [$(date '+%F %T')] 完成 champion seed=${SEED} ##########"
done

echo "########## [$(date '+%F %T')] 开始 kvbench ##########"
bash scripts/run_validation_exp.sh kvbench
echo "########## [$(date '+%F %T')] 完成 kvbench ##########"

echo "全部 P2/P3 队列完成：$(date '+%F %T')"
