#!/bin/bash
# Serial FT chain on the local GPU: Qwen3.5 0.8B x3 seeds, 2B x3, 9B x1,
# all on dataset/ft-all, then generate + score each on the 218 eval prompts.
# Run detached:  setsid bash scripts/ft_scale_chain.sh > results/ft_r2/chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p results/ft_r2 runs

train_eval () {   # model data out cond batch accum extra...
  local model=$1 out=$2 cond=$3 batch=$4 accum=$5 seed=$6; shift 6
  if [ ! -d "runs/$out/adapter" ]; then
    echo "=== train $out (seed $seed) $(date) ==="
    $PY scripts/train_lora.py --model "$model" --data dataset/ft-all \
      --out "runs/$out" --batch "$batch" --accum "$accum" --seed "$seed" "$@" \
      || { echo "TRAIN FAILED $out"; return 1; }
  fi
  if [ ! -f "results/ft_r2/gen_$cond.jsonl" ]; then
    echo "=== generate $cond $(date) ==="
    $PY scripts/batch_generate.py --model "$model" --adapter "runs/$out/adapter" \
      --prompts dataset/ft/eval_prompts.jsonl --out "results/ft_r2/gen_$cond.jsonl" \
      || { echo "GEN FAILED $cond"; return 1; }
  fi
  $PY scripts/ft_eval_io.py score --generations "results/ft_r2/gen_$cond.jsonl" \
      --cond "$cond|L0" || echo "SCORE FAILED $cond"
}

for s in 41 42 43; do
  train_eval Qwen/Qwen3.5-0.8B "ft-r2-08b-s$s" "ft-r2-08b-s$s" 8 2 "$s"
done
for s in 41 42 43; do
  train_eval Qwen/Qwen3.5-2B "ft-r2-2b-s$s" "ft-r2-2b-s$s" 4 4 "$s" --grad-ckpt
done
train_eval Qwen/Qwen3.5-9B "ft-r2-9b-s41" "ft-r2-9b-s41" 2 8 41 --grad-ckpt

echo "=== chain done $(date) ==="
