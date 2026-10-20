#!/bin/bash
# Follow-up chain, run detached:
#   setsid bash scripts/r2_followup_chain.sh > results/ft_r2/chain2.log 2>&1 &
# 1. wait for ft_scale_chain to finish -> Pantheon generations for each model
# 2. wait for resweep to finish -> collect_grid densification (+300 runs)
# 3. rebuild ft dataset (v2, more E cells) -> retrain 0.8B x3 -> eval
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== waiting for ft_scale_chain $(date) ==="
while pgrep -f ft_scale_chain.sh | grep -qv $$; do sleep 60; done
sleep 10

pantheon_gen () {  # cond model [adapter]
  local cond=$1 model=$2 adapter=${3:-}
  [ -f "results/ft_r2/gen_pantheon_$cond.jsonl" ] && return 0
  echo "=== pantheon gen $cond $(date) ==="
  local extra=""
  [ -n "$adapter" ] && extra="--adapter $adapter"
  $PY scripts/batch_generate.py --model "$model" $extra \
    --prompts dataset/pantheon/prompts.jsonl \
    --out "results/ft_r2/gen_pantheon_$cond.jsonl" \
    || { echo "PANTHEON GEN FAILED $cond"; return 1; }
  $PY scripts/pantheon_eval.py \
    --generations "results/ft_r2/gen_pantheon_$cond.jsonl" --cond "$cond"
}

pantheon_gen base-08b Qwen/Qwen3.5-0.8B
pantheon_gen ft-r2-08b-s41 Qwen/Qwen3.5-0.8B runs/ft-r2-08b-s41/adapter
pantheon_gen ft-r2-2b-s41 Qwen/Qwen3.5-2B runs/ft-r2-2b-s41/adapter
pantheon_gen ft-r2-9b-s41 Qwen/Qwen3.5-9B runs/ft-r2-9b-s41/adapter

echo "=== waiting for resweep $(date) ==="
while pgrep -f resweep_reps.py > /dev/null; do sleep 120; done

echo "=== collect_grid densification $(date) ==="
sudo python3 scripts/collect_grid.py --n 300 --seed 7 2>&1 | tail -3

echo "=== rebuild ft dataset v2 $(date) ==="
$PY scripts/build_ft_dataset.py --families ABCDERU --out-dir dataset/ft-v2

for s in 41 42 43; do
  out="ft-v2-08b-s$s"
  if [ ! -d "runs/$out/adapter" ]; then
    echo "=== train $out $(date) ==="
    $PY scripts/train_lora.py --model Qwen/Qwen3.5-0.8B --data dataset/ft-v2 \
      --out "runs/$out" --batch 8 --accum 2 --seed "$s" \
      || { echo "TRAIN FAILED $out"; continue; }
  fi
  $PY scripts/batch_generate.py --model Qwen/Qwen3.5-0.8B \
    --adapter "runs/$out/adapter" --prompts dataset/ft/eval_prompts.jsonl \
    --out "results/ft_r2/gen_$out.jsonl" \
    && $PY scripts/ft_eval_io.py score \
        --generations "results/ft_r2/gen_$out.jsonl" --cond "$out|L0"
done

echo "=== followup chain done $(date) ==="
