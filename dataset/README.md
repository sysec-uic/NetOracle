# dataset/

- `runs/` — the measured run corpus: one JSON per testbed execution
  (config, path, workload, metrics, kernel observables). This backs the
  cells loaded by `eval/data.py`.
- `ft-all/` — the fine-tuning QA set used for the released numbers
  (regenerate: `python3 scripts/build_ft_dataset.py`).
- `pantheon/tasks.jsonl`, `pantheon/prompts.jsonl` — counterfactual
  tasks derived from the public Pantheon archive. Raw archive data is
  not shipped; regenerate with `scripts/pantheon_fetch.py` +
  `scripts/pantheon_tasks.py`.

Ablation dataset variants (ft-U/R/grid/v2/marked) are omitted; each is
reproducible from `runs/` via the flags documented in
`scripts/build_ft_dataset.py` and `scripts/build_marked_assets.py`.

## stackbench/

The 278 StackBench tasks and the 218 leakage-clean task ids, materialized from the frozen matrix; see `dataset/stackbench/README.md` and `scripts/export_stackbench.py`.
