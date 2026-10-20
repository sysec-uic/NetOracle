#!/bin/bash
# Marker-token experiment chain (PF-LLM <load> analog), fully scripted for
# traceability. For each (model, seeds): train on dataset/ft-marked with the
# four marker special tokens, generate on the marked eval prompts, score
# into results/eval/<cond>|L0.jsonl. Unmarked baselines are the existing
# ft-r2-* conditions (identical data minus markers).
#
#   setsid bash scripts/ft_marks_chain.sh > results/ft_r2/marks_chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TOKENS='<anchor>,</anchor>,<target>,</target>'
mkdir -p results/ft_r2 runs

# wait until the GPU is free of the phase daemon / other trainers
while pgrep -f "daemon_phase.py|train_lora.py|batch_generate.py" \
      | grep -qv $$ ; do sleep 60; done

run_one () {  # model out cond batch accum seed extra...
  local model=$1 out=$2 cond=$3 batch=$4 accum=$5 seed=$6; shift 6
  if [ ! -d "runs/$out/adapter" ]; then
    echo "=== train $out (seed $seed) $(date) ==="
    $PY scripts/train_lora.py --model "$model" --data dataset/ft-marked \
      --out "runs/$out" --batch "$batch" --accum "$accum" --seed "$seed" \
      --special-tokens "$TOKENS" "$@" \
      || { echo "TRAIN FAILED $out"; return 1; }
  fi
  if [ ! -f "results/ft_r2/gen_$cond.jsonl" ]; then
    echo "=== generate $cond $(date) ==="
    $PY scripts/batch_generate.py --model "$model" \
      --adapter "runs/$out/adapter" \
      --prompts dataset/ft/eval_prompts_marked.jsonl \
      --out "results/ft_r2/gen_$cond.jsonl" \
      || { echo "GEN FAILED $cond"; return 1; }
  fi
  $PY scripts/ft_eval_io.py score \
      --generations "results/ft_r2/gen_$cond.jsonl" --cond "$cond|L0" \
      || echo "SCORE FAILED $cond"
}

for s in 41 42 43; do
  run_one Qwen/Qwen3.5-0.8B "ft-mk-08b-s$s" "ft-mk-08b-s$s" 8 2 "$s"
done
for s in 41 42 43; do
  run_one Qwen/Qwen3.5-2B "ft-mk-2b-s$s" "ft-mk-2b-s$s" 4 4 "$s" --grad-ckpt
done
# stretch points: one seed each; safe to kill if time is short
run_one Qwen/Qwen3.5-9B "ft-mk-9b-s41" "ft-mk-9b-s41" 2 8 41 --grad-ckpt
run_one google/gemma-4-12B-it "ft-mk-gemma12b-s41" "ft-mk-gemma12b-s41" 1 16 41 --grad-ckpt
# gemma needs an unmarked baseline too (different family, one seed)
if [ ! -d "runs/ft-r2-gemma12b-s41/adapter" ]; then
  echo "=== train ft-r2-gemma12b-s41 (unmarked baseline) $(date) ==="
  $PY scripts/train_lora.py --model google/gemma-4-12B-it --data dataset/ft-all \
    --out runs/ft-r2-gemma12b-s41 --batch 1 --accum 16 --seed 41 --grad-ckpt \
    || echo "TRAIN FAILED gemma unmarked"
fi
if [ -d "runs/ft-r2-gemma12b-s41/adapter" ] && \
   [ ! -f "results/ft_r2/gen_ft-r2-gemma12b-s41.jsonl" ]; then
  $PY scripts/batch_generate.py --model google/gemma-4-12B-it \
    --adapter runs/ft-r2-gemma12b-s41/adapter \
    --prompts dataset/ft/eval_prompts.jsonl \
    --out results/ft_r2/gen_ft-r2-gemma12b-s41.jsonl \
  && $PY scripts/ft_eval_io.py score \
       --generations results/ft_r2/gen_ft-r2-gemma12b-s41.jsonl \
       --cond "ft-r2-gemma12b-s41|L0"
fi

$PY scripts/marks_report.py
echo "=== marks chain done $(date) ==="
