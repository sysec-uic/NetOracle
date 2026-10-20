#!/usr/bin/env python3
"""Standalone plotting for the paper's draft figures.

Self-contained: reads ONLY the JSONs in ./data/ and writes the PDFs into
this folder. No repo dependencies; only matplotlib needed.
Tweak styles/colors/sizes freely and re-run:

  python3 plot_figures.py

Data files (regenerate from the netmodel repo with
scripts/make_figures_hotnets.py / export_grounding.py if the
experiments change):
  data/phase.json     per-epoch daemon trace: epoch, phase, workload,
                      config, goodput_mbps or rr_p99_ms, optional "move"
  data/analysis.json  winners: [(label, count), ...] over n_envs
                      environments; benign: per-seed direction accuracy
                      for effects >2x and <=2x
  data/grounding.json in-context study: per model x level x metric,
                      plus the kNN L2 reference

Colors follow the house palette (8 colors, same hex values as the
palXxx macros in the paper preamble).
"""

from __future__ import annotations

import json
import pathlib
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE  # PDFs land here; copy to overleaf-hotnets/figures/ when done

# ---- style knobs -----------------------------------------------------------
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                     "axes.labelsize": 7.5, "figure.dpi": 200})
# House palette (keep in sync with palXxx macros in the paper preamble).
PAL = {"plum": "#34183E", "blue": "#4D779B", "gray": "#585D5E",
       "magenta": "#82093B", "rose": "#C45C69", "red": "#CD3B42",
       "amber": "#FFC04D", "teal": "#2E8B7F"}
C_MOVE = PAL["red"]                      # move annotations
C_BAND = ["#EDF1F5", "#FFF5E4", "#EAF3F2"]  # light tints: blue/amber/teal
C_BAR = PAL["blue"]                      # primary series
C_BAR2 = PAL["rose"]                     # p99 series
C_MUTED = "#BCBEBE"                      # secondary bar (palGray tint)
PHASE_LABELS = {0: "bulk\n20ms 0.5% 50M", 1: "mixed\n20ms 0% 25M",
                2: "bulk\n40ms 1% 100M"}
# grounding figure: model -> (palette color, marker)
GROUNDING_STYLE = {
    "Gemma4 31B":    (PAL["blue"], "o"),
    "Gemma4 8B":     (PAL["teal"], "s"),
    "GLM-4.7-Flash": (PAL["rose"], "^"),
    "Qwen3.5 35B":   (PAL["magenta"], "D"),
    "Qwen3.5 9.7B":  (PAL["plum"], "v"),
}
LEVEL_LABELS = {"L0": "L0\nzero-shot", "L1": "L1\n+text RAG",
                "L2": "L2\n+exec traces", "L3": "L3\nboth"}
METRIC_TITLES = {"ranking": "Ranking (Kendall's tau)",
                 "counterfactual": "Counterfactual direction acc.",
                 "quantitative": "Quantitative within 1.5x"}


def fig_phase() -> None:
    tr = json.load(open(DATA / "phase.json"))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 2.12), sharex=True,
                                   gridspec_kw={"hspace": 0.12})
    bounds = {}
    for p in sorted({r["phase"] for r in tr}):
        es = [r["epoch"] for r in tr if r["phase"] == p]
        bounds[p] = (min(es) - 0.5, max(es) + 0.5)
    for ax in (ax1, ax2):
        for p, (lo, hi) in bounds.items():
            ax.axvspan(lo, hi, color=C_BAND[p], zorder=0)
    # goodput (bulk epochs), line broken across non-bulk gaps
    pts = [(r["epoch"], r["goodput_mbps"]) for r in tr
           if "goodput_mbps" in r]
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
    # p99 (mixed epochs)
    xs = [r["epoch"] for r in tr if "rr_p99_ms" in r]
    ys = [r["rr_p99_ms"] for r in tr if "rr_p99_ms" in r]
    ax2.plot(xs, ys, "s-", ms=3, lw=1, color=C_BAR2)
    ax2.set_ylabel("p99 (ms)")
    ax2.set_ylim(0, 32)
    ax2.set_xlabel("epoch (10 s each)")
    for r in tr:
        if "move" not in r:
            continue
        ax = ax1 if "goodput_mbps" in r else ax2
        y = r.get("goodput_mbps", r.get("rr_p99_ms"))
        short = r["move"].replace("pfifo_deep", "fifo").replace("/ample", "")
        # keep the red label inside the axes and its arrow short: to the
        # right at the same height for points near the top of the panel
        # (the line drops after a move, so that space is free),
        # above-right otherwise
        lo_y, hi_y = ax.get_ylim()
        near_top = (y - lo_y) > 0.62 * (hi_y - lo_y)
        ax.annotate("move " + short, (r["epoch"], y),
                    textcoords="offset points",
                    xytext=(13, -2) if near_top else (9, 7), fontsize=6,
                    color=C_MOVE,
                    arrowprops=dict(arrowstyle="->", color=C_MOVE, lw=0.7))
    # phase labels live above the top panel, outside the plot area
    for p, (lo, hi) in bounds.items():
        ax1.text((lo + hi) / 2, 1.06, PHASE_LABELS[p], ha="center",
                 va="bottom", fontsize=6, color=PAL["gray"],
                 transform=ax1.get_xaxis_transform())
    fig.savefig(OUT / "fig_phase.pdf", bbox_inches="tight")
    print("fig_phase.pdf")


def fig_analysis() -> None:
    d = json.load(open(DATA / "analysis.json"))
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(3.4, 1.62),
        gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.45})
    # (a) winner frequency (top 8 + rest), bars colored by CC family
    cc_color = {"bbr": PAL["blue"], "cubic": PAL["teal"],
                "illinois": PAL["magenta"], "reno": PAL["rose"]}
    winners = d["winners"]
    top = winners[:8]
    rest = sum(n for _, n in winners[8:])
    names = [n for n, _ in top] + [f"{len(winners) - 8} others"]
    vals = [v for _, v in top] + [rest]
    cols = [cc_color.get(n.split("/")[0], C_MUTED) for n, _ in top] + [C_MUTED]
    ax1.barh(range(len(vals))[::-1], vals, color=cols, height=0.7)
    for i, (v, c) in enumerate(zip(vals, cols)):
        if v >= 5:  # label fits inside the colored block
            ax1.text(v - 0.3, len(vals) - 1 - i, str(v), va="center",
                     ha="right", fontsize=6,
                     color="white" if c != C_MUTED else PAL["gray"])
        else:
            ax1.text(v + 0.2, len(vals) - 1 - i, str(v), va="center",
                     fontsize=6, color=PAL["gray"])
    ax1.set_yticks(range(len(vals))[::-1])
    ax1.set_yticklabels(names, fontsize=6)
    ax1.set_xlabel("environments where optimal")
    ax1.set_title(f"(a) optimal config across\n{d['n_envs']} environments",
                  fontsize=7)
    # (b) direction accuracy by effect size
    big = d["benign"]["acc_big_effect_per_seed"]
    small = d["benign"]["acc_small_effect_per_seed"]
    ax2.bar([0, 1], [mean(small), mean(big)], width=0.55,
            color=[PAL["amber"], PAL["blue"]])
    for x, pts in ((0, small), (1, big)):
        offs = [-0.09, 0.0, 0.09][:len(pts)]
        ax2.plot([x + o for o in offs], pts, "o", ls="none", ms=2.8,
                 mfc="white", mec="#222222", mew=0.7, zorder=3)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["effect\n$\\leq 2\\times$", "effect\n$>2\\times$"],
                        fontsize=6.5)
    ax2.axhline(0.5, ls=":", lw=0.7, color="#888888")
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("direction accuracy")
    ax2.set_title("(b) errors sit on\nnear-ties", fontsize=7)
    fig.savefig(OUT / "fig_analysis.pdf", bbox_inches="tight")
    print("fig_analysis.pdf")


def fig_grounding() -> None:
    d = json.load(open(DATA / "grounding.json"))
    levels = d["levels"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.72))
    for ax, metric in zip(axes, METRIC_TITLES):
        for name, (color, marker) in GROUNDING_STYLE.items():
            per = d["models"].get(name, {})
            xs = [i for i, lv in enumerate(levels)
                  if per.get(lv, {}).get(metric) is not None]
            ys = [per[levels[i]][metric] for i in xs]
            ax.plot(xs, ys, marker=marker, ms=3.2, lw=1.1, color=color,
                    label=name)
        ref = d["knn_L2"].get(metric)
        if ref is not None:
            ax.axhline(ref, ls="--", lw=0.9, color=PAL["gray"],
                       label="kNN (exec only)")
        ax.set_xticks(range(len(levels)),
                      [LEVEL_LABELS[lv] for lv in levels], fontsize=6)
        ax.set_title(METRIC_TITLES[metric], fontsize=7.5)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(alpha=0.25, lw=0.4)
    # one legend row above the panels, outside every axis
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, 1.10),
               columnspacing=1.1, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_grounding.pdf", bbox_inches="tight")
    print("fig_grounding.pdf")


if __name__ == "__main__":
    fig_phase()
    fig_analysis()
    fig_grounding()
