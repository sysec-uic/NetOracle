# StackBench

278 prediction tasks over the Linux network stack, derived from the frozen
evaluation matrix of the execution corpus (run families A to D in
`dataset/runs/`: 456 runs, 228 measured cells). Regenerate with
`python3 scripts/export_stackbench.py`; the output is deterministic.

## Files

- `tasks.jsonl`: one task per line. `task_id` is `<type>:<index>` and matches
  the `task_id` field in every `results/eval/*.jsonl` file.
- `clean_task_ids.json`: the 218 tasks whose target cells are not in the
  fine-tuning set (`dataset/ft-all/`). The paper's Table 1 scores every model
  on exactly these.

## Task types

| type | count | what the model must do | truth |
|---|---|---|---|
| `ranking` | 26 | order 4 to 6 configurations of one environment by `metric` | `better` (`higher` or `lower`), `options` with measured values |
| `counterfactual` | 162 | given measured cell `from_cell`, predict the effect of changing exactly one `knob` to reach `to_cell` | `truth_direction` (`increase`, `decrease`, `same`, with a plus or minus 10% dead band) and `truth_ratio` |
| `quantitative` | 90 | predict `metric` for the unmeasured `target` cell | `truth` |

## Splits

| split | count | rule |
|---|---|---|
| `interp` | 129 | target inside the measured grid; context may use every other cell |
| `heldout_cc` | 116 | every Westwood cell is invisible to context and training; targets use Westwood |
| `heldout_cond` | 47 | targets at 160 ms RTT or 400 Mbit/s, outside anything in context or training (extrapolation) |

Retrieval and training are restricted to a legal pool: no condition ever sees
the answer or a measurement of the target cell (`eval/tasks.py`,
`context_pool`).
