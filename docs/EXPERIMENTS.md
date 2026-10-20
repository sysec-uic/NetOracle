# Experiment map: paper claim → command

Every number in the paper traces to a script here. Commands assume the
repo root as working directory and the shipped `dataset/` + `results/`
in place; scripts that *collect* new data need root and the kernel
modules listed in the README.

## 1. Corpus and testbed

| Claim | Command |
|---|---|
| ~25 s per (workload, config, path) run; frozen matrix 456 runs, corpus 1,800+ | `python3 scripts/collect_grid.py --n 400` (densification); `harness/runner.py` is the per-run engine |
| Noise floor: 0.4–12% CV goodput, <9% latency medians; bimodal tails | `python3 scripts/noise_study.py`; repetitions via `scripts/resweep_reps.py`, analysis via `scripts/resweep_analysis.py` |
| Patch corpus: 170,762 messages / 21,196 commits / 41,758 BM25 units | `crawler/lore_netdev.py`, `crawler/kernel_git.py`, `crawler/build_corpus.py` |
| StackBench: 278 tasks, 3 splits, leakage rules | shipped in `dataset/stackbench/tasks.jsonl` (`python3 scripts/export_stackbench.py` regenerates it from `eval/tasks.py`; `eval/run_eval.py` builds the same tasks in memory) |

## 2. In-context study (§2.3; Figure 1, Table 1, F1–F4)

| Claim | Command |
|---|---|
| 5 open-weight models × L0–L3 evidence levels | `python3 eval/run_eval.py --backend ollama:<model> --levels L0,L1,L2,L3` (set `OLLAMA_HOST`) |
| kNN / per-task GBT baselines given L2 evidence | `python3 eval/baselines.py`; sensitivity: `python3 scripts/knn_sensitivity.py` |
| Aggregates and Figure 1 | `python3 eval/analyze.py`; plot: `figures/export_grounding.py` + `figures/plot_figures.py` |

## 3. Fine-tuned world model (§4; Table 2)

| Claim | Command |
|---|---|
| FT dataset: 726 leakage-filtered cells → ~3.8k QA pairs | `python3 scripts/build_ft_dataset.py` |
| 20-min LoRA fine-tune (rank 32, 12.8M params) | `python3 scripts/train_lora.py` |
| 0.8B/2B/9B scale sweep, 3 seeds each | `bash scripts/ft_scale_chain.sh` |
| Generation on the 218-task clean subset | `python3 scripts/batch_generate.py`; I/O helpers in `scripts/ft_eval_io.py` |
| Table 2 numbers + paired-bootstrap CIs | `python3 scripts/eval_stats.py` |
| GBT/RF trained on identical cells | `python3 scripts/regressor_baseline.py`; pooled variant `scripts/pool_regressor.py` |
| Marker-token ablation (negative result) | `scripts/marks.py`, `scripts/build_marked_assets.py`, `bash scripts/ft_marks_chain.sh`, report `scripts/marks_report.py` |

## 4. Flywheel (§4)

| Claim | Command |
|---|---|
| 92% surprise precision vs. random decaying to ~55% | `python3 scripts/flywheel.py` (v2 with live testbed: `scripts/flywheel_v2.py`) |
| Spread ranks errors (ρ=0.59–0.63) but covers only 47% | `python3 scripts/calibration.py` |

## 5. Gated policy (§5; Table 3, Figure 5)

| Claim | Command |
|---|---|
| 1–3 measurements to optimum-or-tie on ground-truth sweeps | `python3 scripts/policy_hillclimb.py`; full-space ground truth via `scripts/reachability.py` |
| Safety-ingredient ladder (Table 3) | `python3 scripts/ablation_gate.py` |
| Bayesian-optimization / model-free replay (4–6 and 15–19) | `python3 scripts/bo_replay.py` |
| Phase-tracking daemon (Figure 5) | `sudo -E python3 scripts/daemon_phase.py`; single-env demo `scripts/daemon_demo.py` |
| Real-WAN bufferbloat: p99 409→43 ms | `python3 scripts/realpath_climb.py` (cloud sender → campus receiver; see script header) |
| Policy scoring across environments | `python3 scripts/policy_eval.py` |

## 6. External validation (§5, Pantheon)

| Claim | Command |
|---|---|
| 2,153 counterfactual tasks from 652 public real-path experiments | `python3 scripts/pantheon_fetch.py` then `python3 scripts/pantheon_tasks.py` (shipped: `dataset/pantheon/tasks.jsonl`) |
| 0.36–0.41 direction vs. 0.57 prior; cellular-Vegas 0.92 | `python3 scripts/pantheon_eval.py` |

## 7. Figures

All paper figures regenerate from `figures/` (see `figures/README.md`);
`scripts/make_figures_hotnets.py` refreshes the plot-ready JSON from
`results/`.

## 8. Camera-ready additions (generalization outside the testbed)

| Claim | Command / shipped output |
|---|---|
| Endpoint observables only (goodput, smoothed RTT, retransmits, queue drops and backlog) lose nothing on the testbed | `results/eval/ft-U-r3-s4*\|L0.jsonl` (observables-only models) vs `results/eval/ft-r3-08b-s4*\|L0.jsonl`, scored with `python3 scripts/eval_stats.py`; `results/pantheon_replay/score_observables_only.json` |
| Emulation-only model on 15 Pantheon paths: 0.37 vs 0.35 for always "no change" | `python3 scripts/pantheon_diet.py train-base`, `prompts`, `score`; aggregate in `results/pantheon_replay/cvp/score_cvagg.json` (`base`, `nochange`) |
| Replaying other real paths in six node folds: 0.68 direction, policy at 0.94 of optimum vs 0.76 for CUBIC, RF 0.70 | `PANTHEON_FOLD=k PANTHEON_NFOLDS=6 python3 scripts/pantheon_diet.py replay --round 1` then `train_lora.py` and `score`, for k in 0..5; shipped: `results/pantheon_replay/cv/`, `results/pantheon_replay/cvp/`, `results/pantheon_replay/score_baselines.json` |
| Larger fine-tuned models do worse off-grid on a later corpus (0.67 / 0.60 / 0.63 for 0.8B / 2B / 9B) | `results/eval/ft-r3-{08b,2b,9b}-s4*\|L0.jsonl`, scored with `python3 scripts/eval_stats.py`; the later corpus itself is not part of this release |
| Unseen knobs (sfq, pie): every real effect identified without retraining on 16 measured pairs | `sudo -E python3 scripts/newknob_zeroshot.py --stage all`; shipped: `results/newknob/` |
| Competing CUBIC flows: competitors within -18% to +5% goodput on 7 of 8 lossy paths, RTT cut 60-88% on bloated paths | `sudo python3 scripts/coexist.py --reps 3`; shipped: `results/coexist/` |
