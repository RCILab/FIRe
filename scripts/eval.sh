#!/bin/bash
set -u
GPU=$1; PORT=$2; NAME=$3; MODEL=$4; TASK=$5; RES=$6
FIRE=$(cd "$(dirname "$0")/.." && pwd); OUT=$FIRE/eval; LOG=$FIRE/logs/eval; mkdir -p $OUT $LOG
exec 9>$LOG/.gpu${GPU}.lock; flock 9
source $FIRE/scripts/vla_server.sh
echo "server up $PORT"
bash -c "source $CONDA_SH && conda activate ${ISAACLAB_ENV:-isaaclab} && cd $FIRE && CUDA_VISIBLE_DEVICES=$GPU VLA_PORT=$PORT PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES timeout 7200 python scripts/play.py --task=$TASK --headless --enable_cameras --checkpoint=$RES --seed ${SEED:-42} --num_envs 50 --save_actions --max_steps ${MAX_STEPS:-1200} env.vla_model=$VLA ${ENV_ARGS:-}" > $LOG/play_$NAME.log 2>&1 < /dev/null
csv=$(grep -oE "/[^ ]*actions_[0-9_]+\.csv" $LOG/play_$NAME.log | tail -1)
if grep -q "Actions saved" $LOG/play_$NAME.log && [ -f "$csv" ]; then mv "$csv" $OUT/$NAME.csv; echo "DONE $OUT/$NAME.csv"; else echo "FAILED (see $LOG/play_$NAME.log)"; fi
stop_vla_server
