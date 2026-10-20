#!/usr/bin/env python3
"""Draft (placeholder-quality) figures for the HotNets draft.

  fig_phase.pdf     heartbeat trace of the phase-tracking daemon
                    (results/policy/daemon_phase.json)
  fig_analysis.pdf  (a) frequency of the optimal configuration across
                    corpus environments; (b) counterfactual direction
                    accuracy by true effect size (3 seeds)

Outputs land in results/ and are copied by the caller into the paper's
figures/ directory. Deliberately simple styling; intended to be redrawn
by hand for camera-ready.

  .venv/bin/python scripts/make_figures_hotnets.py
"""

from __future__ import annotations

import collections
import json
import math
import pathlib
import sys
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "results"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                     "axes.labelsize": 7.5, "figure.dpi": 200})

C_MOVE = "#c23b22"
C_BAND = ["#eef3fb", "#fdf3e7", "#eef7ee"]
C_BAR = "#4878a8"


def fig_phase() -> None:
    tr = json.load(open(REPO / "results" / "policy" / "daemon_phase.json"))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.4, 2.4), sharex=True,
        gridspec_kw={"hspace": 0.12})
    phases = sorted({r["phase"] for r in tr})
    bounds = {}
    for p in phases:
        es = [r["epoch"] for r in tr if r["phase"] == p]
        bounds[p] = (min(es) - 0.5, max(es) + 0.5)
    labels = {0: "bulk 20ms/0.5%/50M", 1: "mixed 20ms/0%/25M",
              2: "bulk 40ms/1%/100M"}
    for ax in (ax1, ax2):
        for p, (lo, hi) in bounds.items():
            ax.axvspan(lo, hi, color=C_BAND[p], zorder=0)
    # goodput panel (bulk epochs); break the line across non-bulk gaps
    pts = [(r["epoch"], r["goodput_mbps"]) for r in tr if "goodput_mbps" in r]
    seg = [[pts[0]]]
    for prev, cur in zip(pts, pts[1:]):
        if cur[0] - prev[0] > 1:
            seg.append([])
        seg[-1].append(cur)
    for s in seg:
        ax1.plot([x for x, _ in s], [y for _, y in s], "o-", ms=3, lw=1,
                 color=C_BAR)
    ax1.set_ylabel("goodput (Mbit/s)")
    ax1.set_ylim(0, 105)
    # p99 panel (mixed epochs)
    xs2 = [r["epoch"] for r in tr if "rr_p99_ms" in r]
    ys2 = [r["rr_p99_ms"] for r in tr if "rr_p99_ms" in r]
    ax2.plot(xs2, ys2, "s-", ms=3, lw=1, color="#a05195")
    ax2.set_ylabel("p99 (ms)")
    ax2.set_ylim(0, 32)
    ax2.set_xlabel("epoch (10 s each)")
    # annotate moves
    for r in tr:
        if "move" not in r:
            continue
        ax = ax1 if "goodput_mbps" in r else ax2
        y = r.get("goodput_mbps", r.get("rr_p99_ms"))
        short = r["move"].replace("pfifo_deep", "fifo").replace(
            "/ample", "").replace("fq_codel", "fq_codel")
        ax.annotate("move " + short, (r["epoch"], y),
                    textcoords="offset points", xytext=(6, 8),
                    fontsize=6, color=C_MOVE,
                    arrowprops=dict(arrowstyle="->", color=C_MOVE, lw=0.7))
    for p, (lo, hi) in bounds.items():
        ax1.text((lo + hi) / 2, 100, labels[p], ha="center", va="top",
                 fontsize=6, color="#555555")
    fig.savefig(OUT / "fig_phase.pdf", bbox_inches="tight")
    print("fig_phase.pdf")


def env_winners() -> collections.Counter:
    cells: dict = collections.defaultdict(dict)
    for f in (REPO / "dataset" / "runs").glob("*.json"):
        r = json.loads(f.read_text())
        c = r["config"]
        if c["family"] not in "ABCDERUP":
            continue
        env = (c["workload"], c["rtt_ms"], c["loss_pct"], c["bw_mbit"])
        key = (c["cc"], c["qdisc"], c["buf"])
        try:
            v = (r["bulk"]["recv_mbps"] if c["workload"] == "bulk" else
                 r["rr"]["lat_p99_ms"] if c["workload"] in ("rr", "mixed")
                 else r["short"]["fct_p95_ms"])
        except KeyError:
            continue
        cells[env].setdefault(key, []).append(v)
    winners = collections.Counter()
    for env, d in cells.items():
        if len(d) < 5:
            continue
        hi = env[0] == "bulk"
        agg = {k: mean(v) for k, v in d.items()}
        best = (max if hi else min)(agg, key=agg.get)
        label = "/".join([best[0], best[1].replace("pfifo_", "")])
        if best[2] == "limited":
            label += "/ltd"
        winners[label] += 1
    return winners


def benign_points() -> tuple[list, list]:
    sys.path.insert(0, str(REPO / "eval"))
    sys.path.insert(0, str(HERE))
    from build_ft_dataset import committed_cells  # noqa: E402
    from tasks import build_all  # noqa: E402
    cells = committed_cells()
    tasks = build_all(cells)
    idx = {f"{t}:{i}": task for t, l in tasks.items()
           for i, task in enumerate(l)}
    ids = set(json.load(open(REPO / "dataset" / "ft" / "manifest.json"))
              ["ft_eval_task_ids"])
    big_acc, small_acc = [], []
    for seed in (41, 42, 43):
        rows = [json.loads(l) for l in
                open(REPO / "results" / "eval" / f"ft-all-r0-s{seed}|L0.jsonl")
                if json.loads(l)["task_id"] in ids]
        big, small = [], []
        for r in rows:
            if r["type"] != "counterfactual":
                continue
            task = idx[r["task_id"]]
            va = task["from_cell"]["primary_value"]
            tr = task.get("ratio") or (
                task["to_cell"]["primary_value"] / va if va else None)
            if not tr or tr <= 0:
                continue
            ok = r["scores"].get("dir_correct") == 1
            (big if abs(math.log10(tr)) > math.log10(2) else small).append(ok)
        big_acc.append(sum(big) / len(big))
        small_acc.append(sum(small) / len(small))
    return big_acc, small_acc


def fig_analysis() -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(3.4, 1.7),
        gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.45})
    # (a) winner frequency
    winners = env_winners()
    top = winners.most_common(8)
    rest = sum(winners.values()) - sum(n for _, n in top)
    names = [n for n, _ in top] + [f"{len(winners) - 8} others"]
    vals = [v for _, v in top] + [rest]
    ax1.barh(range(len(vals))[::-1], vals, color=C_BAR, height=0.7)
    for i, v in enumerate(vals):
        ax1.text(v + 0.1, len(vals) - 1 - i, str(v), va="center", fontsize=6)
    ax1.set_yticks(range(len(vals))[::-1])
    ax1.set_yticklabels(names, fontsize=6)
    ax1.set_xlabel("environments where optimal")
    ax1.set_title(f"(a) optimal config across\n"
                  f"{sum(winners.values())} environments", fontsize=7)
    # (b) direction accuracy by effect size
    big, small = benign_points()
    ax2.bar([0, 1], [mean(small), mean(big)], width=0.55,
            color=["#b0b8c4", C_BAR])
    for x, pts in ((0, small), (1, big)):
        ax2.plot([x] * len(pts), pts, "k.", ms=4)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["effect\n$\\leq 2\\times$", "effect\n$>2\\times$"],
                        fontsize=6.5)
    ax2.axhline(0.5, ls=":", lw=0.7, color="#888888")
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("direction accuracy")
    ax2.set_title("(b) errors sit on\nnear-ties", fontsize=7)
    fig.savefig(OUT / "fig_analysis.pdf", bbox_inches="tight")
    print("fig_analysis.pdf")


if __name__ == "__main__":
    fig_phase()
    fig_analysis()
