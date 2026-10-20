#!/bin/bash
# A/B acquisition experiment: grid-only baseline, then random arm, then
# uncertainty arm. Arms are data-isolated (each trains on ABCD + own family).
set -e
cd "$(dirname "$0")/.."

SEEDS=41,42,43
ROUNDS=4
PER_ROUND=24

echo "=== BASELINE (grid-only, seeds $SEEDS) ==="
python3 scripts/flywheel_v2.py --baseline-only --seeds $SEEDS

echo "=== ARM R (random) ==="
python3 scripts/flywheel_v2.py --arm R --strategy random \
    --rounds $ROUNDS --per-round $PER_ROUND --seeds $SEEDS --seed 7

echo "=== ARM U (uncertainty) ==="
python3 scripts/flywheel_v2.py --arm U --strategy uncertainty \
    --rounds $ROUNDS --per-round $PER_ROUND --seeds $SEEDS --seed 7

echo "=== AB DONE ==="
