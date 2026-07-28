#!/usr/bin/env bash
# $1 = mode (mtp_on|mtp_off), $2 = port, $3 = binary dir, $4 = logfile
set -u
MODE="$1"; PORT="$2"; BINDIR="$3"; LOG="$4"
MODEL=/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/snapshots/f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf
DRAFT=/home/ianblenke/.cache/kaggle_mtp_head_upload/mtp-gemma-4-31B-it-Q8_0.gguf
ARGS=(--model "$MODEL" -ngl 999 --ctx-size 32768
      --cache-type-k q8_0 --cache-type-v q8_0
      --host 127.0.0.1 --port "$PORT" --no-warmup)
if [ "$MODE" = "mtp_on" ]; then
  ARGS+=(--spec-type draft-mtp --model-draft "$DRAFT")
fi
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$BINDIR"
exec "$BINDIR/llama-server" "${ARGS[@]}" > "$LOG" 2>&1
