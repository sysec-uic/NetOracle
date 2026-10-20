"""Load raw harness runs and aggregate them into experimental cells.

A *cell* is the mean over repetitions of one (workload, stack config, path
condition) point, carrying both workload-level metrics and kernel-side
observables. Cells are the atoms the world-model tasks are built from.
"""

from __future__ import annotations

import json
import pathlib
import statistics
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
RUNS_DIR = HERE.parent / "dataset" / "runs"

CONFIG_KEYS = ["family", "workload", "cc", "qdisc", "rtt_ms", "loss_pct", "bw_mbit", "buf"]


def _run_metrics(rec: dict) -> dict:
    """Flatten one run's metrics of interest."""
    out: dict = {}
    if "bulk" in rec:
        b = rec["bulk"]
        out["goodput_mbps"] = b["recv_mbps"]
        out["sent_mbps"] = b["goodput_mbps"]
        out["retransmits"] = b.get("retransmits") or 0
    if "rr" in rec:
        r = rec["rr"]
        out["rr_p50_ms"] = r["lat_p50_ms"]
        out["rr_p99_ms"] = r["lat_p99_ms"]
        out["rr_rps"] = r["rps"]
    if "short" in rec:
        s = rec["short"]
        out["fct_p50_ms"] = s["fct_p50_ms"]
        out["fct_p95_ms"] = s["fct_p95_ms"]
    ss = rec.get("ss_mid") or {}
    if "cwnd" in ss:
        out["cwnd_mid"] = ss["cwnd"]
    if "srtt_ms" in ss:
        out["srtt_mid_ms"] = ss["srtt_ms"]
    q = rec.get("qdisc") or {}
    if "qdisc_dropped" in q:
        out["qdisc_drops"] = q["qdisc_dropped"]
    qm = rec.get("qdisc_mid") or {}
    if qm.get("qdisc_backlog_bytes"):
        out["qdisc_backlog_kb_mid"] = round(qm["qdisc_backlog_bytes"] / 1e3, 1)
    return out


# What "better" means, per workload, for ranking/counterfactual tasks.
PRIMARY_METRIC = {
    "bulk": ("goodput_mbps", "higher"),
    "rr": ("rr_p99_ms", "lower"),
    "mixed": ("rr_p99_ms", "lower"),  # latency under load == bufferbloat exposure
    "short": ("fct_p95_ms", "lower"),
}


def load_cells(runs_dir: pathlib.Path = RUNS_DIR) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for f in sorted(runs_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        cfg = rec["config"]
        key = tuple(cfg[k] for k in CONFIG_KEYS)
        groups[key].append(_run_metrics(rec))

    cells = []
    for key, runs in groups.items():
        cfg = dict(zip(CONFIG_KEYS, key))
        metrics: dict = {}
        for m in {k for r in runs for k in r}:
            vals = [r[m] for r in runs if m in r]
            metrics[m] = round(statistics.mean(vals), 3)
            if len(vals) > 1:
                metrics[m + "_std"] = round(statistics.stdev(vals), 3)
        pm, direction = PRIMARY_METRIC[cfg["workload"]]
        cells.append({
            **cfg,
            "n_reps": len(runs),
            "metrics": metrics,
            "primary_metric": pm,
            "primary_value": metrics.get(pm),
            "better": direction,
        })
    # family S = smoke-test runs; not part of the dataset proper
    return [c for c in cells if c["primary_value"] is not None and c["family"] != "S"]


def config_of(cell: dict) -> dict:
    return {k: cell[k] for k in CONFIG_KEYS}


def serialize_cell(cell: dict, with_metrics: bool = True) -> str:
    """Compact human/LLM-readable record."""
    cfg = (f"workload={cell['workload']} cc={cell['cc']} qdisc={cell['qdisc']} "
           f"rtt={cell['rtt_ms']}ms loss={cell['loss_pct']}% bw={cell['bw_mbit']}Mbit "
           f"buf={cell['buf']}")
    if not with_metrics:
        return cfg
    keep = ["goodput_mbps", "retransmits", "rr_p50_ms", "rr_p99_ms", "fct_p50_ms",
            "fct_p95_ms", "cwnd_mid", "srtt_mid_ms", "qdisc_drops", "qdisc_backlog_kb_mid"]
    ms = " ".join(f"{k}={cell['metrics'][k]}" for k in keep if k in cell["metrics"])
    return f"{cfg} -> {ms}"


def config_distance(a: dict, b: dict) -> float:
    """Similarity metric for picking related-execution context (L2)."""
    d = 0.0
    d += 0.0 if a["workload"] == b["workload"] else 2.0
    d += 0.0 if a["cc"] == b["cc"] else 1.0
    d += 0.0 if a["qdisc"] == b["qdisc"] else 1.0
    d += 0.0 if a["buf"] == b["buf"] else 1.0
    d += abs(a["rtt_ms"] - b["rtt_ms"]) / 80.0
    d += abs(a["loss_pct"] - b["loss_pct"])
    d += abs(a["bw_mbit"] - b["bw_mbit"]) / 200.0
    return d


if __name__ == "__main__":
    cells = load_cells()
    by_fam: dict[str, int] = defaultdict(int)
    for c in cells:
        by_fam[c["family"]] += 1
    print(f"{len(cells)} cells: {dict(by_fam)}")
    for c in cells[:3]:
        print(serialize_cell(c))
