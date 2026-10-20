"""Produce paper-ready tables and figures from eval results + the dataset.

Outputs into results/:
  table_main.md / table_main.tex     grounding-level x task-type headline table
  table_splits.md                    generalization splits (interp vs held-out)
  fig_grounding.pdf                  metric vs grounding level, per model
  fig_splits.pdf                     held-out generalization bars (LLM vs kNN)
  fig_bufferbloat.pdf                dataset case study: qdisc vs rr p99 under load
  fig_cc_matrix.pdf                  dataset heatmap: CC x (rtt,loss) goodput

Run with the project venv: ../.venv/bin/python analyze.py
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_cells

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
EVAL_DIR = RESULTS / "eval"

LEVELS = ["L0", "L1", "L2", "L3"]
LEVEL_LABELS = {"L0": "L0\nzero-shot", "L1": "L1\n+patch/doc RAG",
                "L2": "L2\n+exec traces", "L3": "L3\nboth"}
HEADLINE = {
    "ranking": ("tau", "Ranking (Kendall's tau)"),
    "counterfactual": ("dir_correct", "Counterfactual direction acc."),
    "quantitative": ("within_1_5x", "Quantitative within 1.5x"),
}


def load_rows() -> dict[str, list[dict]]:
    out = {}
    for f in sorted(EVAL_DIR.glob("*.jsonl")):
        out[f.stem] = [json.loads(l) for l in f.read_text().splitlines()]
    return out


def cond_parts(cond: str) -> tuple[str, str]:
    """'cli-haiku|L2' -> ('cli-haiku', 'L2'); knn -> ('knn', 'L2')."""
    model, _, level = cond.partition("|")
    return model, level


def mean_of(rows: list[dict], field: str) -> float | None:
    vals = [r["scores"].get(field) for r in rows
            if isinstance(r["scores"].get(field), (int, float))]
    return sum(vals) / len(vals) if vals else None


def fmt(v: float | None) -> str:
    return "--" if v is None else f"{v:.2f}"


def table_main(rows_by_cond: dict) -> str:
    models = sorted({cond_parts(c)[0] for c in rows_by_cond})
    lines = ["| Model | Level | Rank tau | Rank top-1 | CF dir acc | CF log-ratio err | Quant <1.5x | Quant MAPE |",
             "|---|---|---|---|---|---|---|---|"]
    for model in models:
        levels = ["L2"] if model == "knn" else LEVELS
        for level in levels:
            cond = f"{model}|{level}"
            if cond not in rows_by_cond:
                continue
            rows = rows_by_cond[cond]
            by_type = defaultdict(list)
            for r in rows:
                by_type[r["type"]].append(r)
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                model, level,
                fmt(mean_of(by_type["ranking"], "tau")),
                fmt(mean_of(by_type["ranking"], "top1")),
                fmt(mean_of(by_type["counterfactual"], "dir_correct")),
                fmt(mean_of(by_type["counterfactual"], "log_ratio_err")),
                fmt(mean_of(by_type["quantitative"], "within_1_5x")),
                fmt(mean_of(by_type["quantitative"], "ape")),
            ))
    return "\n".join(lines)


def table_splits(rows_by_cond: dict) -> str:
    lines = ["| Model | Level | Task | interp | heldout_cc | heldout_cond |",
             "|---|---|---|---|---|---|"]
    for cond in sorted(rows_by_cond):
        model, level = cond_parts(cond)
        rows = rows_by_cond[cond]
        for ttype, (field, _) in HEADLINE.items():
            by_split = defaultdict(list)
            for r in rows:
                if r["type"] == ttype:
                    by_split[r["split"]].append(r)
            if not by_split:
                continue
            lines.append("| {} | {} | {} | {} | {} | {} |".format(
                model, level, ttype,
                fmt(mean_of(by_split.get("interp", []), field)),
                fmt(mean_of(by_split.get("heldout_cc", []), field)),
                fmt(mean_of(by_split.get("heldout_cond", []), field)),
            ))
    return "\n".join(lines)


def fig_grounding(rows_by_cond: dict) -> None:
    models = sorted({cond_parts(c)[0] for c in rows_by_cond if cond_parts(c)[0] != "knn"})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for ax, (ttype, (field, title)) in zip(axes, HEADLINE.items()):
        for model in models:
            ys, xs = [], []
            for i, level in enumerate(LEVELS):
                rows = rows_by_cond.get(f"{model}|{level}", [])
                v = mean_of([r for r in rows if r["type"] == ttype], field)
                if v is not None:
                    xs.append(i)
                    ys.append(v)
            ax.plot(xs, ys, marker="o", label=model.replace("cli-", ""))
        knn_rows = rows_by_cond.get("knn|L2", [])
        v = mean_of([r for r in knn_rows if r["type"] == ttype], field)
        if v is not None:
            ax.axhline(v, ls="--", color="gray", lw=1, label="kNN (exec only)")
        ax.set_xticks(range(4), [LEVEL_LABELS[l] for l in LEVELS], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_grounding.pdf")
    fig.savefig(RESULTS / "fig_grounding.png", dpi=150)
    plt.close(fig)


def fig_splits(rows_by_cond: dict) -> None:
    """Held-out generalization: best-grounded LLM vs kNN, per split."""
    conds = [c for c in rows_by_cond if c.endswith("|L3")] + ["knn|L2"]
    conds = [c for c in conds if c in rows_by_cond]
    splits = ["interp", "heldout_cc", "heldout_cond"]
    ttype, field = "counterfactual", "dir_correct"
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.8 / max(1, len(conds))
    for i, cond in enumerate(conds):
        rows = [r for r in rows_by_cond[cond] if r["type"] == ttype]
        by_split = defaultdict(list)
        for r in rows:
            by_split[r["split"]].append(r)
        vals = [mean_of(by_split.get(s, []), field) or 0 for s in splits]
        label = cond_parts(cond)[0].replace("cli-", "") + (
            " (L3)" if cond.endswith("L3") else " (exec only)")
        ax.bar([x + i * width for x in range(len(splits))], vals, width, label=label)
    ax.set_xticks([x + width * (len(conds) - 1) / 2 for x in range(len(splits))],
                  ["interpolation", "held-out CC", "held-out condition"])
    ax.set_ylabel("counterfactual direction accuracy")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_splits.pdf")
    fig.savefig(RESULTS / "fig_splits.png", dpi=150)
    plt.close(fig)


def fig_bufferbloat(cells: list[dict]) -> None:
    """Dataset case study: latency-under-load across qdiscs (family B)."""
    fam = [c for c in cells if c["family"] == "B" and c["workload"] == "mixed"
           and c["bw_mbit"] == 25]
    if not fam:
        return
    qdiscs = ["pfifo_deep", "pfifo_bdp", "fq", "fq_codel", "cake"]
    ccs = sorted({c["cc"] for c in fam})
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.8 / len(ccs)
    for i, cc in enumerate(ccs):
        vals = []
        for q in qdiscs:
            m = [c for c in fam if c["cc"] == cc and c["qdisc"] == q]
            vals.append(m[0]["metrics"].get("rr_p99_ms", 0) if m else 0)
        ax.bar([x + i * width for x in range(len(qdiscs))], vals, width, label=cc)
    ax.set_xticks([x + width * (len(ccs) - 1) / 2 for x in range(len(qdiscs))], qdiscs,
                  fontsize=8)
    ax.set_ylabel("RR p99 latency under load (ms)")
    ax.set_title("Bufferbloat: 25 Mbit bottleneck, 20 ms RTT", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_bufferbloat.pdf")
    fig.savefig(RESULTS / "fig_bufferbloat.png", dpi=150)
    plt.close(fig)


def fig_cc_matrix(cells: list[dict]) -> None:
    """Dataset heatmap: bulk goodput, CC x (rtt, loss)."""
    fam = [c for c in cells if c["family"] == "A" and c["workload"] == "bulk"]
    if not fam:
        return
    ccs = sorted({c["cc"] for c in fam})
    conds = sorted({(c["rtt_ms"], c["loss_pct"]) for c in fam})
    grid = [[next((c["metrics"]["goodput_mbps"] for c in fam
                   if c["cc"] == cc and (c["rtt_ms"], c["loss_pct"]) == cond), float("nan"))
             for cond in conds] for cc in ccs]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(ccs)), ccs, fontsize=8)
    ax.set_xticks(range(len(conds)), [f"{r}ms\n{l}%" for r, l in conds], fontsize=7)
    ax.set_xlabel("RTT / loss", fontsize=8)
    fig.colorbar(im, label="goodput (Mbit/s)")
    ax.set_title("Bulk TCP goodput, 100 Mbit bottleneck", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_cc_matrix.pdf")
    fig.savefig(RESULTS / "fig_cc_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    rows_by_cond = load_rows()
    cells = load_cells()
    if rows_by_cond:
        main_tbl = table_main(rows_by_cond)
        splits_tbl = table_splits(rows_by_cond)
        (RESULTS / "table_main.md").write_text(main_tbl + "\n")
        (RESULTS / "table_splits.md").write_text(splits_tbl + "\n")
        print(main_tbl)
        print()
        print(splits_tbl)
        fig_grounding(rows_by_cond)
        fig_splits(rows_by_cond)
    fig_bufferbloat(cells)
    fig_cc_matrix(cells)
    print(f"\nfigures + tables written to {RESULTS}")


if __name__ == "__main__":
    main()
