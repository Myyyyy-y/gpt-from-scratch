#!/bin/bash
# P1 GPU queue: run the four P1 trainings in sequence on one GPU (~8h).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_queue_gpu.sh
#   PYTHON=/path/to/cuda-env/bin/python CUDA_VISIBLE_DEVICES=7 bash scripts/run_queue_gpu.sh

set -euo pipefail

for NAME in combo zeroinit untie qknorm_clean; do
    echo "########## [$(date '+%F %T')] start ${NAME} ##########"
    bash scripts/run_validation_exp.sh "${NAME}"
    echo "########## [$(date '+%F %T')] done ${NAME} ##########"
done
echo "P1 queue finished: $(date '+%F %T')"
