#!/usr/bin/env python3
"""Data Flywheel: Active Learning loop for the Network World Model.

This script samples unknown configurations, uses an LLM to predict their
outcome, and then executes them on the testbed. Highly surprising outcomes
(where the LLM prediction diverges significantly from reality) are logged
for future fine-tuning.

Usage:
  python3 scripts/flywheel.py --backend cli:haiku --steps 5 --metric goodput_mbps
"""

import argparse
import glob
import json
import pathlib
import random
import sys
import time

# Ensure we can import from harness and eval
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(HERE.parent / "harness"))

import runner
import topology as topo
from config_space import RunConfig, mutate_config
from llm import make_backend
from prompts import quantitative_prompt
from data import load_cells

def get_actual_metric(result: dict, metric: str) -> float:
    """Extract a simple scalar metric from a testbed result dictionary."""
    if metric == "goodput_mbps" and "bulk" in result:
        return result["bulk"].get("goodput_mbps", 0.0)
    if metric == "mean_rtt_us" and "bulk" in result:
        return result["bulk"].get("mean_rtt_us", 0.0)
    # Fallback to rr metric if it's an rr workload
    if metric == "p99_rtt_ms" and "rr" in result:
        return result["rr"]["percentiles"].get("99.0", 0.0)
    raise ValueError(f"Metric {metric} not found in result or not supported yet.")

def get_candidate(cells: list[dict], existing_ids: set[str]) -> RunConfig:
    """Mutate an existing cell's config to create a novel RunConfig."""
    base_cell = random.choice(cells)
    base_cfg_dict = {k: base_cell[k] for k in ["family", "workload", "cc", "qdisc", "rtt_ms", "loss_pct", "bw_mbit", "buf"]}
    base_cfg = RunConfig(**base_cfg_dict)
    for _ in range(50):
        # Mutate 1-2 knobs
        candidate = mutate_config(base_cfg, n_mutations=random.choice([1, 2]))
        if candidate.cell_id not in existing_ids:
            return candidate
    # Fallback if somehow stuck
    return mutate_config(base_cfg, 3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cli:haiku", help="LLM backend (e.g. cli:haiku)")
    ap.add_argument("--steps", type=int, default=5, help="Number of flywheel iterations")
    ap.add_argument("--metric", default="goodput_mbps", help="Metric to predict and evaluate")
    ap.add_argument("--threshold", type=float, default=0.2, help="Error threshold (20%%) to save as hard example")
    ap.add_argument("--dry-run", action="store_true", help="Don't run actual testbed execution (for debugging)")
    args = ap.parse_args()

    runs_dir = HERE.parent / "dataset" / "runs"
    flywheel_log = HERE.parent / "dataset" / "flywheel_hard_examples.jsonl"
    
    print("Loading existing dataset...")
    cells = load_cells(runs_dir)
    # The cell identity from config_space is cell_id, which we can compute by making a RunConfig
    existing_ids = set()
    for c in cells:
        cfg_dict = {k: c[k] for k in ["family", "workload", "cc", "qdisc", "rtt_ms", "loss_pct", "bw_mbit", "buf"]}
        existing_ids.add(RunConfig(**cfg_dict).cell_id)
    print(f"Loaded {len(cells)} cells.")

    llm = make_backend(args.backend)
    print(f"Initialized LLM backend: {llm.name}")

    if not args.dry_run:
        topo.create()
        servers = runner.Servers()
        servers.start_rr()
        print("Testbed topology created and servers started.")

    try:
        for step in range(args.steps):
            print(f"\n--- Flywheel Step {step + 1}/{args.steps} ---")
            
            # 1. Generate candidate
            candidate = get_candidate(cells, existing_ids)
            print(f"Generated candidate config: {candidate.describe()}")

            # 2. Setup Task for Prediction
            task = {
                "type": "quantitative",
                "split": "interp",
                "metric": args.metric,
                "target": candidate.describe(),
                # Base context on L2 (measured runs) and L1 (RAG docs)
            }

            # 3. Predict with World Model (LLM)
            prompt = quantitative_prompt(task, "L2", cells)
            print(f"Querying LLM for prediction on {args.metric}...")
            
            predicted_val = None
            try:
                prediction_json = llm.query_json(prompt, retries=1)
                if prediction_json and "value" in prediction_json:
                    predicted_val = float(prediction_json["value"])
                    reasoning = prediction_json.get("reasoning", "")
                    print(f"LLM Predicted {args.metric}: {predicted_val} (Reason: {reasoning})")
                else:
                    print("LLM failed to return a valid numeric prediction. Skipping.")
                    continue
            except Exception as e:
                print(f"LLM query error: {e}. Skipping.")
                continue

            # 4. Act (Execute in testbed)
            if args.dry_run:
                print("[Dry-run] Skipping testbed execution.")
                actual_val = predicted_val * random.uniform(0.5, 1.5) # Dummy value
            else:
                print(f"Executing on testbed (run_id: {candidate.run_id})...")
                try:
                    result = runner.execute(candidate, servers)
                    # Save the run
                    (runs_dir / f"{candidate.run_id}.json").write_text(json.dumps(result, indent=1))
                    existing_ids.add(candidate.cell_id)
                    # Reload cells so context pool can use the new run
                    cells = load_cells(runs_dir)
                    
                    actual_val = get_actual_metric(result, args.metric)
                    print(f"Testbed Actual {args.metric}: {actual_val}")
                except Exception as e:
                    print(f"Execution failed: {e}. Skipping.")
                    continue

            # 5. Observe & Learn (Calculate Error)
            if actual_val > 0:
                error = abs(predicted_val - actual_val) / actual_val
            else:
                error = abs(predicted_val - actual_val) # Absolute error if actual is 0
                
            print(f"Prediction Error: {error:.1%}")

            if error > args.threshold:
                print(f"[*] SURPRISE! Error > {args.threshold:.1%}. Saving to hard examples.")
                log_entry = {
                    "ts": time.time(),
                    "config": candidate.describe(),
                    "metric": args.metric,
                    "predicted": predicted_val,
                    "actual": actual_val,
                    "error": error,
                    "reasoning": reasoning
                }
                with open(flywheel_log, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            else:
                print("Model prediction was accurate. Not logging as hard example.")
                
    finally:
        if not args.dry_run:
            servers.stop_all()
            topo.destroy()
            print("Testbed cleaned up.")

if __name__ == "__main__":
    main()
