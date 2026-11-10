#!/bin/bash
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
GR00T_DIR=${GR00T_DIR:-$HOME/Isaac-GR00T}; OPENPI_DIR=${OPENPI_DIR:-$HOME/openpi}; OPENVLA_DIR=${OPENVLA_DIR:-$HOME/openvla-oft}
VLA=${VLA:-gr00t}
case $VLA in
  gr00t) SERVER="source $CONDA_SH && conda activate gr00t && cd $GR00T_DIR && PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU exec python scripts/inference_service.py --server --model_path $MODEL --embodiment-tag franka --data-config franka_triple_cam_eefframe --denoising-steps 4 --port $PORT" ;;
  pi05) SERVER="cd $OPENPI_DIR && PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=${PI05_MEM_FRACTION:-0.5} exec .venv/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config=${PI05_CONFIG:-pi05_franka} --policy.dir=$MODEL" ;;
  openvla) SERVER="source $CONDA_SH && conda activate ${OPENVLA_ENV:-openvla-oft} && cd $OPENVLA_DIR && PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU exec python vla-scripts/deploy_batch.py --pretrained_checkpoint $MODEL --port $PORT ${OPENVLA_ARGS:-}" ;;
  *) echo "unknown VLA=$VLA (gr00t|pi05|openvla)"; exit 1 ;;
esac
if [ "$MODEL" != external ]; then
  setsid bash -c "$SERVER" > $LOG/server_$NAME.log 2>&1 < /dev/null &
fi
for _ in $(seq 1 120); do ss -ltn | grep -q ":$PORT " && break; sleep 5; done
stop_vla_server() {
  [ "$MODEL" = external ] && return
  sp=$(ps -eo pid,args | awk -v p="--port $PORT" '(index($0,"inference_service.py") || index($0,"serve_policy.py") || index($0,"deploy_batch.py")) && index($0,p) && !/awk/ {print $1}'); [ -n "$sp" ] && kill $sp
}
