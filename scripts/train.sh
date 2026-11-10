#!/bin/bash
set -u
GPU=$1; PORT=$2; TASK=$3; MODEL=$4; NAME=$5; RES=${6:-}
FIRE=$(cd "$(dirname "$0")/.." && pwd); LOG=$FIRE/logs/train; mkdir -p $LOG
source $FIRE/scripts/vla_server.sh
source $CONDA_SH && conda activate ${ISAACLAB_ENV:-isaaclab} && cd $FIRE
CUDA_VISIBLE_DEVICES=$GPU VLA_PORT=$PORT PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES python scripts/train.py --task=FIRe-$TASK-v0 \
  --headless --enable_cameras ${RES:+--checkpoint=$RES} ${SAVE_BEST:+--save_best=$SAVE_BEST} env.vla_model=$VLA ${ENV_ARGS:-} 2>&1 | tee $LOG/train_$NAME.log
echo "EXIT ${PIPESTATUS[0]}" | tee -a $LOG/train_$NAME.log
stop_vla_server
