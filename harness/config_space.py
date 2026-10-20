"""Configuration space for the network-stack world-model dataset.

A *run* is one execution of a workload under a fully specified stack
configuration and emulated path condition.  Runs are grouped into scenario
families, each designed to expose a different mechanism of the kernel
network stack:

  A  cc-path      congestion control x (rtt, loss) sweep      -> CC dynamics
  B  qdisc        AQM/bufferbloat sweep at the bottleneck      -> queueing
  C  buffer       socket-buffer-limited transfers              -> flow control
  D  frontier     off-grid conditions (unseen rtt / bw values) -> generalization
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass, field

CC_ALGOS = ["cubic", "bbr", "reno", "vegas", "westwood", "illinois"]
QDISCS = ["pfifo_deep", "pfifo_bdp", "fq_codel", "cake", "fq"]
WORKLOADS = ["bulk", "rr", "mixed", "short"]

# sysctl tcp_wmem/tcp_rmem (min default max) presets
BUF_PRESETS = {
    "ample": {"wmem": "4096 131072 33554432", "rmem": "4096 262144 33554432"},
    "limited": {"wmem": "4096 65536 262144", "rmem": "4096 65536 262144"},
}


@dataclass(frozen=True)
class RunConfig:
    family: str
    workload: str
    cc: str
    qdisc: str
    rtt_ms: int
    loss_pct: float
    bw_mbit: int
    buf: str  # "ample" | "limited"
    rep: int = 0

    @property
    def cell_id(self) -> str:
        """Identity of the experimental cell (all reps of a cell share it)."""
        d = asdict(self)
        d.pop("rep")
        blob = json.dumps(d, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    @property
    def run_id(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    @property
    def bdp_pkts(self) -> int:
        """Bandwidth-delay product in 1500B packets (>= 4)."""
        bdp_bytes = self.bw_mbit * 1e6 / 8 * self.rtt_ms / 1e3
        return max(4, int(bdp_bytes / 1500))

    def describe(self) -> dict:
        d = asdict(self)
        d["run_id"] = self.run_id
        d["cell_id"] = self.cell_id
        return d


def mutate_config(cfg: RunConfig, n_mutations: int = 1,
                  family: str = "F") -> RunConfig:
    """Generate a counterfactual neighbor by mutating one or more fields randomly."""
    d = asdict(cfg)
    mutations = random.sample(["cc", "qdisc", "rtt_ms", "loss_pct", "bw_mbit", "buf"], min(n_mutations, 6))
    for m in mutations:
        if m == "cc":
            d["cc"] = random.choice([x for x in CC_ALGOS if x != d["cc"]])
        elif m == "qdisc":
            d["qdisc"] = random.choice([x for x in QDISCS if x != d["qdisc"]])
        elif m == "rtt_ms":
            # e.g., multiply or divide by 2, or pick from common grid [10, 20, 40, 80, 160]
            d["rtt_ms"] = random.choice([10, 20, 40, 80, 160])
        elif m == "loss_pct":
            d["loss_pct"] = random.choice([0.0, 0.1, 1.0])
        elif m == "bw_mbit":
            d["bw_mbit"] = random.choice([25, 50, 100, 400])
        elif m == "buf":
            d["buf"] = "limited" if d["buf"] == "ample" else "ample"
    
    d["family"] = family  # flywheel arm tag ("F"; "U"/"R" for A/B arms)
    d["rep"] = 0
    return RunConfig(**d)



def family_a(reps: int) -> list[RunConfig]:
    """CC x path-condition sweep: the core counterfactual grid."""
    out = []
    for cc, rtt, loss, wl, rep in itertools.product(
        CC_ALGOS, [10, 40, 80], [0.0, 0.1, 1.0], ["bulk", "rr", "mixed"], range(reps)
    ):
        out.append(RunConfig("A", wl, cc, "pfifo_deep", rtt, loss, 100, "ample", rep))
    return out


def family_b(reps: int) -> list[RunConfig]:
    """Bottleneck qdisc sweep: bufferbloat / AQM interactions."""
    out = []
    for qd, cc, bw, wl, rep in itertools.product(
        QDISCS, ["cubic", "bbr"], [25, 100], ["mixed", "short"], range(reps)
    ):
        out.append(RunConfig("B", wl, cc, qd, 20, 0.0, bw, "ample", rep))
    return out


def family_c(reps: int) -> list[RunConfig]:
    """Socket-buffer-limited bulk transfers (rwnd/wmem-capped throughput)."""
    out = []
    for buf, cc, rtt, rep in itertools.product(
        ["ample", "limited"], ["cubic", "bbr"], [10, 40, 80], range(reps)
    ):
        out.append(RunConfig("C", "bulk", cc, "pfifo_deep", rtt, 0.0, 100, buf, rep))
    return out


def family_d(reps: int) -> list[RunConfig]:
    """Off-grid conditions used only as held-out evaluation targets."""
    out = []
    for cc, loss, wl, rep in itertools.product(
        ["cubic", "bbr", "vegas"], [0.0, 1.0], ["bulk", "mixed"], range(reps)
    ):
        out.append(RunConfig("D", wl, cc, "pfifo_deep", 160, loss, 100, "ample", rep))
    for cc, wl, rep in itertools.product(["cubic", "bbr"], ["bulk"], range(reps)):
        out.append(RunConfig("D", wl, cc, "pfifo_deep", 20, 0.0, 400, "ample", rep))
    return out


FAMILIES = {"A": family_a, "B": family_b, "C": family_c, "D": family_d}


def enumerate_matrix(families: str = "ABCD", reps: int = 2) -> list[RunConfig]:
    out: list[RunConfig] = []
    for f in families:
        out.extend(FAMILIES[f](reps))
    return out


def smoke_matrix() -> list[RunConfig]:
    """One run per workload type for a quick end-to-end check."""
    return [
        RunConfig("S", "bulk", "cubic", "pfifo_deep", 20, 0.0, 100, "ample"),
        RunConfig("S", "rr", "bbr", "fq_codel", 20, 0.0, 100, "ample"),
        RunConfig("S", "mixed", "cubic", "pfifo_deep", 20, 0.0, 25, "ample"),
        RunConfig("S", "short", "bbr", "cake", 20, 0.0, 100, "ample"),
        RunConfig("S", "bulk", "cubic", "pfifo_deep", 40, 0.0, 100, "limited"),
    ]


if __name__ == "__main__":
    m = enumerate_matrix()
    by_fam = {}
    for r in m:
        by_fam[r.family] = by_fam.get(r.family, 0) + 1
    print(f"total runs: {len(m)}  by family: {by_fam}")
    est = len(m) * 16 / 3600
    print(f"rough wall-clock estimate @16s/run: {est:.1f} h")
