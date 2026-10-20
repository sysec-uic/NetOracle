"""Experiment runner: executes the config matrix on the netns dumbbell.

One JSON record per run lands in --out (default ../dataset/runs).  Runs are
identified by a content hash of their config, so re-invocation resumes where
it left off.  Usage:

  python3 runner.py --smoke
  python3 runner.py --families ABCD --reps 2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import subprocess
import threading
import time

import topology as topo
from config_space import RunConfig, enumerate_matrix, smoke_matrix
from metrics import (
    counter_delta,
    parse_proc_counters,
    parse_qdisc_stats,
    parse_ss_best,
    qdisc_delta,
)

HERE = pathlib.Path(__file__).resolve().parent
RR_TOOL = HERE / "rr_tool.py"
IPERF_PORT = 5201
RR_PORT = 5599
BULK_SECS = 10
RR_SECS = 8
MIXED_SECS = 12


def sh_out(cmd: str, timeout: int = 90) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {cmd}\n{r.stderr[:2000]}")
    return r.stdout


class Servers:
    """Persistent rr server; per-run one-off iperf3 servers."""

    def __init__(self) -> None:
        self.rr_proc: subprocess.Popen | None = None

    def start_rr(self) -> None:
        self.rr_proc = subprocess.Popen(
            f"sudo ip netns exec {topo.NS['sv']} python3 {RR_TOOL} server --port {RR_PORT}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)

    def start_iperf_oneoff(self) -> None:
        subprocess.run(f"sudo pkill -f 'iperf3 -s -1 -p {IPERF_PORT}'", shell=True,
                       capture_output=True)
        subprocess.Popen(
            f"sudo ip netns exec {topo.NS['sv']} iperf3 -s -1 -p {IPERF_PORT}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)

    def stop_all(self) -> None:
        subprocess.run("sudo pkill -f 'rr_tool.py server'", shell=True, capture_output=True)
        subprocess.run("sudo pkill -f 'iperf3 -s'", shell=True, capture_output=True)


def mid_run_snapshot(result: dict, delay: float, ss_filter: str) -> threading.Timer:
    def snap() -> None:
        try:
            result["ss_mid"] = parse_ss_best(
                sh_out(topo.ns_exec("cl", f"ss -tin {ss_filter}"), timeout=10)
            )
            result["qdisc_mid"] = parse_qdisc_stats(topo.qdisc_stats())
        except Exception as e:  # snapshot is best-effort
            result["snap_error"] = str(e)

    t = threading.Timer(delay, snap)
    t.start()
    return t


def run_iperf(cfg: RunConfig, servers: Servers, secs: int) -> dict:
    servers.start_iperf_oneoff()
    out = sh_out(
        topo.ns_exec(
            "cl",
            f"iperf3 -c {topo.SERVER_IP} -p {IPERF_PORT} -t {secs} -O 2 -J "
            f"-C {cfg.cc}",
        ),
        timeout=secs + 45,
    )
    j = json.loads(out)
    end = j["end"]
    intervals = [
        round(i["sum"]["bits_per_second"] / 1e6, 2)
        for i in j.get("intervals", [])
        if not i["sum"].get("omitted")
    ]
    return {
        "goodput_mbps": round(end["sum_sent"]["bits_per_second"] / 1e6, 3),
        "recv_mbps": round(end["sum_received"]["bits_per_second"] / 1e6, 3),
        "retransmits": end["sum_sent"].get("retransmits"),
        "interval_mbps": intervals,
        "max_rtt_us": end["streams"][0]["sender"].get("max_rtt"),
        "mean_rtt_us": end["streams"][0]["sender"].get("mean_rtt"),
    }


def run_rr(secs: float) -> dict:
    out = sh_out(
        topo.ns_exec(
            "cl",
            f"python3 {RR_TOOL} rr-client --host {topo.SERVER_IP} --port {RR_PORT} "
            f"--duration {secs}",
        ),
        timeout=int(secs) + 60,
    )
    return json.loads(out)


def run_short(flows: int = 40, size: int = 200_000) -> dict:
    out = sh_out(
        topo.ns_exec(
            "cl",
            f"python3 {RR_TOOL} flow-client --host {topo.SERVER_IP} --port {RR_PORT} "
            f"--flows {flows} --size {size}",
        ),
        timeout=240,
    )
    return json.loads(out)


def execute(cfg: RunConfig, servers: Servers) -> dict:
    topo.apply(cfg)
    rec: dict = {"config": cfg.describe(), "ts": time.time()}
    before_proc = parse_proc_counters(topo.proc_netstat("cl"))
    before_q = parse_qdisc_stats(topo.qdisc_stats())
    snap: threading.Timer | None = None

    if cfg.workload == "bulk":
        snap = mid_run_snapshot(rec, BULK_SECS * 0.6, f"dport = :{IPERF_PORT}")
        rec["bulk"] = run_iperf(cfg, servers, BULK_SECS)
    elif cfg.workload == "rr":
        rec["rr"] = run_rr(RR_SECS)
    elif cfg.workload == "mixed":
        snap = mid_run_snapshot(rec, MIXED_SECS * 0.5, f"dport = :{IPERF_PORT}")
        bulk_res: dict = {}
        err: list[str] = []

        def bulk_thread() -> None:
            try:
                bulk_res.update(run_iperf(cfg, servers, MIXED_SECS))
            except Exception as e:
                err.append(str(e))

        t = threading.Thread(target=bulk_thread)
        t.start()
        time.sleep(2.5)  # let bulk exit slow start before measuring rr
        rec["rr"] = run_rr(RR_SECS)
        t.join(timeout=MIXED_SECS + 60)
        if err:
            raise RuntimeError(f"mixed bulk failed: {err[0]}")
        rec["bulk"] = bulk_res
    elif cfg.workload == "short":
        rec["short"] = run_short()
    else:
        raise ValueError(cfg.workload)

    if snap:
        snap.join()
    after_proc = parse_proc_counters(topo.proc_netstat("cl"))
    after_q = parse_qdisc_stats(topo.qdisc_stats())
    rec["counters"] = counter_delta(before_proc, after_proc)
    rec["qdisc"] = qdisc_delta(before_q, after_q)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="ABCD")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(HERE.parent / "dataset" / "runs"))
    ap.add_argument("--limit", type=int, default=0, help="stop after N new runs")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    matrix = smoke_matrix() if args.smoke else enumerate_matrix(args.families, args.reps)
    random.Random(42).shuffle(matrix)
    todo = [c for c in matrix if not (out / f"{c.run_id}.json").exists()]
    print(f"matrix={len(matrix)} done={len(matrix) - len(todo)} todo={len(todo)}", flush=True)

    topo.create()
    servers = Servers()
    servers.start_rr()
    done = 0
    t_start = time.time()
    try:
        for i, cfg in enumerate(todo):
            if args.limit and done >= args.limit:
                break
            label = (f"{cfg.family}/{cfg.workload} cc={cfg.cc} qdisc={cfg.qdisc} "
                     f"rtt={cfg.rtt_ms} loss={cfg.loss_pct} bw={cfg.bw_mbit} "
                     f"buf={cfg.buf} rep={cfg.rep}")
            for attempt in (1, 2):
                try:
                    rec = execute(cfg, servers)
                    (out / f"{cfg.run_id}.json").write_text(json.dumps(rec, indent=1))
                    done += 1
                    elapsed = time.time() - t_start
                    eta_min = elapsed / done * (len(todo) - i - 1) / 60
                    print(f"[{i + 1}/{len(todo)}] ok {label} (eta {eta_min:.0f}m)", flush=True)
                    break
                except Exception as e:
                    print(f"[{i + 1}/{len(todo)}] attempt {attempt} FAILED {label}: {e}",
                          flush=True)
                    if attempt == 2:
                        (out / f"{cfg.run_id}.error").write_text(str(e))
                    else:
                        subprocess.run(f"sudo pkill -f 'iperf3 -c'", shell=True,
                                       capture_output=True)
                        servers.stop_all()
                        topo.create()
                        servers.start_rr()
    finally:
        servers.stop_all()
        topo.destroy()
    print(f"finished: {done} new runs in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
