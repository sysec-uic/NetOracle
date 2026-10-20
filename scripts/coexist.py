#!/usr/bin/env python3
"""Coexistence of NetOracle's moves with competing CUBIC flows at a shared bottleneck.

The gated policy makes two kinds of move: cubic -> bbr on lossy bulk paths (queue
unchanged), and deep FIFO -> fq_codel on mixed workloads (cubic kept). For each we run
one controlled flow and N competing cubic flows through the testbed bottleneck, before
and after the move, and record what the competitors get. A third block is the boundary
the policy does not cross: bbr against cubic on a clean path, deep and shallow buffer.

  python3 scripts/coexist.py [--reps 3]   -> results/coexist/runs.jsonl, summary.json
"""
from __future__ import annotations
import argparse, json, pathlib, statistics as st, subprocess, sys, time
HERE = pathlib.Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(REPO / "harness"))
import topology as topo  # noqa: E402
from config_space import RunConfig  # noqa: E402

OUT = REPO / "results" / "coexist"
PORT0 = 5301

# (block, env name, rtt, loss, bw, [N competitors], before (cc, qdisc), after (cc, qdisc))
PLAN = [
    ("lossy", "S00", 20, 0.5, 25, [1], ("cubic", "pfifo_deep"), ("bbr", "pfifo_deep")),
    ("lossy", "S01", 40, 1.0, 100, [1, 3], ("cubic", "pfifo_deep"), ("bbr", "pfifo_deep")),
    ("lossy", "S02", 60, 2.0, 50, [1], ("cubic", "pfifo_deep"), ("bbr", "pfifo_deep")),
    ("lossy", "S03", 80, 0.5, 200, [1, 3], ("cubic", "pfifo_deep"), ("bbr", "pfifo_deep")),
    ("lossy", "S04", 120, 1.0, 75, [1], ("cubic", "pfifo_deep"), ("bbr", "pfifo_deep")),
    ("lossy", "S05", 160, 0.5, 100, [1], ("cubic", "pfifo_deep"), ("bbr", "pfifo_deep")),
    ("queue", "S12", 10, 0.0, 100, [1, 3], ("cubic", "pfifo_deep"), ("cubic", "fq_codel")),
    ("queue", "S13", 20, 0.0, 200, [1], ("cubic", "pfifo_deep"), ("cubic", "fq")),
    ("queue", "S15", 80, 0.0, 75, [1], ("cubic", "pfifo_deep"), ("cubic", "fq_codel")),
    ("boundary", "clean_deep", 40, 0.0, 100, [1], ("cubic", "pfifo_deep"), ("bbr", "pfifo_deep")),
    ("boundary", "clean_shallow", 40, 0.0, 100, [1], ("cubic", "pfifo_bdp"), ("bbr", "pfifo_bdp")),
]


def run(rtt, loss, bw, qdisc, ccs, secs=25, omit=5) -> list[dict]:
    topo.apply(RunConfig("X", "bulk", "cubic", qdisc, rtt, loss, bw, "ample"))
    subprocess.run("sudo pkill -f '[i]perf3 -s -1'", shell=True, capture_output=True)
    for i in range(len(ccs)):
        subprocess.Popen(f"sudo ip netns exec {topo.NS['sv']} iperf3 -s -1 -p {PORT0 + i}", shell=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    procs = [subprocess.Popen(topo.ns_exec("cl", f"iperf3 -c {topo.SERVER_IP} -p {PORT0 + i} -t {secs} -O {omit} -J -C {cc}"),
                              shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for i, cc in enumerate(ccs)]
    flows = []
    for cc, p in zip(ccs, procs):
        out, _ = p.communicate(timeout=secs + omit + 60)
        try:
            e = json.loads(out)["end"]
            flows.append({"cc": cc, "mbps": round(e["sum_received"]["bits_per_second"] / 1e6, 3),
                          "retrans": e["sum_sent"].get("retransmits"),
                          "mean_rtt_ms": round(e["streams"][0]["sender"].get("mean_rtt", 0) / 1e3, 2)})
        except Exception as ex:  # noqa: BLE001
            flows.append({"cc": cc, "error": str(ex)[:80]})
    return flows


def jain(xs):
    return sum(xs) ** 2 / (len(xs) * sum(x * x for x in xs)) if xs and sum(xs) > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    topo.create()
    with open(OUT / "runs.jsonl", "a") as f:
        for rep in range(a.reps):
            for block, env, rtt, loss, bw, ns, before, after in PLAN:
                for n in ns:
                    for phase, (cc, qd) in (("before", before), ("after", after)):
                        flows = run(rtt, loss, bw, qd, [cc] + ["cubic"] * n)
                        row = {"block": block, "env": env, "rtt": rtt, "loss": loss, "bw": bw, "n_comp": n,
                               "phase": phase, "controlled_cc": cc, "qdisc": qd, "rep": rep, "flows": flows}
                        f.write(json.dumps(row) + "\n"); f.flush()
                        print(env, n, phase, [(x.get("cc"), x.get("mbps"), x.get("mean_rtt_ms")) for x in flows], flush=True)
    topo.destroy()
    summarize()


def summarize() -> None:
    rows = [json.loads(l) for l in open(OUT / "runs.jsonl")]
    rows = [r for r in rows if all("mbps" in x for x in r["flows"])]
    keys = sorted({(r["block"], r["env"], r["n_comp"]) for r in rows})
    out = []
    for block, env, n in keys:
        rec = {"block": block, "env": env, "n_comp": n}
        for phase in ("before", "after"):
            rr = [r for r in rows if (r["block"], r["env"], r["n_comp"], r["phase"]) == (block, env, n, phase)]
            if not rr:
                continue
            rec[phase] = {"controlled_cc": rr[0]["controlled_cc"], "qdisc": rr[0]["qdisc"], "bw": rr[0]["bw"],
                          "controlled_mbps": round(st.median(r["flows"][0]["mbps"] for r in rr), 2),
                          "competitor_mbps": round(st.median(st.mean(x["mbps"] for x in r["flows"][1:]) for r in rr), 2),
                          "competitor_rtt_ms": round(st.median(st.mean(x["mean_rtt_ms"] for x in r["flows"][1:]) for r in rr), 1),
                          "jain": round(st.median(jain([x["mbps"] for x in r["flows"]]) for r in rr), 3),
                          "total_mbps": round(st.median(sum(x["mbps"] for x in r["flows"]) for r in rr), 2), "reps": len(rr)}
        if "before" in rec and "after" in rec:
            b, af = rec["before"], rec["after"]
            rec["competitor_change_pct"] = round(100 * (af["competitor_mbps"] / b["competitor_mbps"] - 1), 1) if b["competitor_mbps"] else None
            rec["competitor_rtt_change_pct"] = round(100 * (af["competitor_rtt_ms"] / b["competitor_rtt_ms"] - 1), 1) if b["competitor_rtt_ms"] else None
        out.append(rec)
        print(json.dumps(rec), flush=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    summarize() if len(sys.argv) > 1 and sys.argv[1] == "summarize" else main()
