#!/bin/bash
# P2/P3 GPU queue: wait for a free GPU, then run the remaining GPU tasks in sequence (~10h).
#
# Order: layernorm ablation -> data-size ablation -> AttnRes training
#        -> champion x3 seeds -> 29M GPU KV-cache re-test
#
# Usage:
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_queue_phase2.sh

set -euo pipefail

GPU_ID="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"

wait_gpu_free() {
    echo ">>> [$(date '+%F %T')] waiting for GPU ${GPU_ID} to be free ..."
    while :; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                -i "${GPU_ID}" 2>/dev/null | head -1)
        if [ -n "${used}" ] && [ "${used}" -lt 500 ]; then
            echo ">>> [$(date '+%F %T')] GPU ${GPU_ID} free (${used} MiB); starting queue"
            return
        fi
        sleep 120
    done
}

wait_gpu_free

for NAME in layernorm data3m attnres; do
    echo "########## [$(date '+%F %T')] start ${NAME} ##########"
    bash scripts/run_validation_exp.sh "${NAME}"
    echo "########## [$(date '+%F %T')] done ${NAME} ##########"
done

for SEED in 1 2 3; do
    echo "########## [$(date '+%F %T')] start champion seed=${SEED} ##########"
    SEED="${SEED}" bash scripts/run_validation_exp.sh champion
    echo "########## [$(date '+%F %T')] done champion seed=${SEED} ##########"
done

echo "########## [$(date '+%F %T')] start kvbench ##########"
bash scripts/run_validation_exp.sh kvbench
echo "########## [$(date '+%F %T')] done kvbench ##########"

echo "P2/P3 queue finished: $(date '+%F %T')"
