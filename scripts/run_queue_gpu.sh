#!/bin/bash
# P1 GPU 队列：在单卡上按序跑完 P1 四项训练（约 8 小时）
#
# 用法：
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_queue_gpu.sh
#   PYTHON=/path/to/cuda-env/bin/python CUDA_VISIBLE_DEVICES=7 bash scripts/run_queue_gpu.sh

set -euo pipefail

for NAME in combo zeroinit untie qknorm_clean; do
    echo "########## [$(date '+%F %T')] 开始 ${NAME} ##########"
    bash scripts/run_validation_exp.sh "${NAME}"
    echo "########## [$(date '+%F %T')] 完成 ${NAME} ##########"
done
echo "全部 P1 队列完成：$(date '+%F %T')"
