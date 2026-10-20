# NetOracle

**A world model for the Linux kernel network stack**: a small language
model, fine-tuned on counterfactual testbed executions, that predicts
what a configuration change (congestion control, qdisc, socket buffers)
will do to a workload's performance — before anyone touches the machine.
A confidence-gated policy turns those predictions into configuration
moves that reach near-optimal settings in 1–3 live measurements and
never end below the kernel default.

> Research prototype accompanying our HotNets '26 paper.
> Not production software; the testbed reshapes network namespaces and
> requires root.

## How it works

```
                 offline (emulated testbed)          online (live host)
┌──────────────┐   runs    ┌───────────────┐  anchor  ┌──────────────┐
│ data engine  ├──────────►│  world model  │◄─────────┤ live traffic │
│ netns dumbbell│          │ 0.8B LM + LoRA│          └──────┬───────┘
│ ~25 s per run│◄──────────┤ (this repo)   ├─────────►┌──────▼───────┐
└──────────────┘ next-batch└───────────────┘ predicts │ gate: move   │
                 (flywheel)                           │ or hold      │
                                                      └──────────────┘
```

1. **Data engine** (`harness/`): a four-namespace dumbbell on Linux —
   HTB bottleneck + configurable child qdisc on one router, netem
   delay/loss on the other. One (workload, configuration, path) point
   runs unattended in ~25 s.
2. **Task distillation** (`eval/tasks.py`, `scripts/build_ft_dataset.py`):
   runs become quantitative and counterfactual QA pairs with strict
   JSON targets and leakage filtering.
3. **World model** (`scripts/train_lora.py`): LoRA fine-tuning of a
   0.8B LM on those pairs (~20 min on one consumer GPU).
4. **Flywheel** (`scripts/flywheel*.py`): the model's sampling spread
   picks the next configurations to measure; retrain and repeat.
5. **Gated policy** (`scripts/policy_hillclimb.py`, `daemon_*.py`):
   anchored on one live measurement, move one knob only if ≥4/5
   sampled predictions agree on a ≥15% win; otherwise hold.

## Repository map

| Path | What it is |
|---|---|
| `harness/` | netns dumbbell testbed (topology, runner, config space, metrics) |
| `eval/` | StackBench task construction, scoring, LLM backends, baselines |
| `scripts/` | all experiment entry points (see `docs/EXPERIMENTS.md`) |
| `crawler/` | patch-corpus builders (netdev archive, kernel git) |
| `figures/` | paper-figure reproduction kit (matplotlib + JSON) |
| `dataset/` | run corpus (`runs/`), fine-tuning set (`ft-all/`), Pantheon tasks |
| `results/` | evaluation outputs backing every table/figure in the paper |
| `docs/EXPERIMENTS.md` | paper claim → command map |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Reproduce the paper's tables and figures from shipped results (no
hardware needed):**

```bash
python3 scripts/eval_stats.py          # Table 2 + significance tests
python3 scripts/ablation_gate.py --replay  # Table 3 safety ladder
cd figures && python3 plot_figures.py  # Figures 1, 4, 5
```

**Collect your own runs (requires root + Linux with bbr/vegas/... modules):**

```bash
sudo modprobe -a tcp_bbr tcp_vegas tcp_westwood tcp_illinois
python3 scripts/collect_grid.py --n 50 --seed 1
```

**Train the world model (one GPU, ~20 min):**

```bash
python3 scripts/build_ft_dataset.py
python3 scripts/train_lora.py --base Qwen/Qwen3.5-0.8B --data dataset/ft-all
```

**Evaluate:**

```bash
python3 eval/run_eval.py --backend hf:runs/<adapter> --levels L0
python3 scripts/eval_stats.py
```

**Run the gated daemon on a live host:**

```bash
sudo -E python3 scripts/daemon_demo.py     # single-environment demo
sudo -E python3 scripts/daemon_phase.py    # three-phase tracking (Fig. 5)
```

## Requirements

- Linux ≥ 6.x with `tcp_bbr`, `tcp_vegas`, `tcp_westwood`,
  `tcp_illinois` modules; `iproute2`, `iperf3`; root for netns setup.
  (Inside containers, `net.ipv4.tcp_allowed_congestion_control` is
  owned by the host.)
- Python ≥ 3.10; GPU optional (needed only for training/inference;
  Ollama/OpenAI-compatible backends supported via `OLLAMA_HOST`).

## Data and models

- `dataset/runs/` — the measured run corpus (frozen matrix + growth).
- `dataset/ft-all/` — the fine-tuning set behind the released numbers.
- `dataset/pantheon/tasks.jsonl` — 2,153 counterfactual tasks derived
  from the public Pantheon archive (regenerate with
  `scripts/pantheon_fetch.py` + `scripts/pantheon_tasks.py`).
- Model weights are not in this repo (size); training is cheap enough
  to reproduce from scratch with the commands above.

## License

MIT — see `LICENSE`.

## Citation

Luyao Bai and Xiaoguang Wang. Toward a World Model for the Linux Kernel
Network Stack. In *Proceedings of the 25th ACM Workshop on Hot Topics in
Networks (HotNets '26)*, Salt Lake City, UT, USA, November 2026.

```bibtex
@inproceedings{netoracle2026,
  author    = {Luyao Bai and Xiaoguang Wang},
  title     = {Toward a World Model for the Linux Kernel Network Stack},
  booktitle = {Proceedings of the 25th ACM Workshop on Hot Topics in Networks (HotNets '26)},
  year      = {2026},
  address   = {Salt Lake City, UT, USA},
  publisher = {ACM}
}
```
