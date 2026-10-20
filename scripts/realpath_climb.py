#!/usr/bin/env python3
"""Real-path (sim-to-real) validation of the gated hill-climb chooser.

The tuned host is a cloud sender (a public-cloud VM (e.g., Chameleon Cloud)) shipping data to a campus
campus receiver over the real Internet via tailscale (WireGuard/UDP — the
inner TCP sees the real path). The sender applies (cc, qdisc, buf); a
configurable HTB/cake bottleneck at the sender egress makes queueing real;
RTT, jitter, loss and cross-traffic are the live WAN's. No netem anywhere.

The model was trained ONLY on the emulated netns testbed; this script asks
whether its knowledge transfers to a real path (unseen rtt=31ms).

  python3 scripts/realpath_climb.py

Outputs results/policy/realpath.json
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(HERE.parent / "harness"))

from config_space import BUF_PRESETS, RunConfig  # noqa: E402
from flywheel_v2 import knobs_of  # noqa: E402
from policy_hillclimb import gated_best, neighbors, predict_neighbors  # noqa: E402

OUT_DIR = HERE.parent / "results" / "policy"

CH = "129.114.108.85"        # management ssh (real NIC, not tailscale)
CH_TS = "100.73.56.42"       # chameleon on the tailnet (sender)
LOCAL_TS = "100.91.127.104"  # this box on the tailnet (receiver)
DEV = "tailscale0"           # sender egress carrying the experiment traffic
RTT_MS = 31                  # measured base RTT of the real path

MAX_STEPS = 3
# (workload, bottleneck_mbit): mixed = bulk + rr sharing the sender egress.
# rates sit below tailscaled's ~22 Mbps forwarded-traffic ceiling so OUR
# middlebox stays the bottleneck; 15/8 are values the model never trained on
ENVS_W = [("mixed", 15), ("mixed", 8), ("bulk", 15)]
PRIMARY = {"mixed": ("rr_p99_ms", "lower"), "bulk": ("goodput_mbps", "higher")}


def sh(cmd: str, timeout: int = 60) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {cmd}\n{r.stderr[:300]}")
    return r.stdout


def ch(cmd: str, timeout: int = 60) -> str:
    return sh(f"ssh -o BatchMode=yes {CH} {json.dumps(cmd)}", timeout)


def bdp_pkts(bw_mbit: int) -> int:
    return max(4, int(bw_mbit * 1e6 / 8 * RTT_MS / 1e3 / 1280))


def setup_middlebox() -> None:
    """Client netns -> middlebox netns (bottleneck qdisc) -> root -> real WAN.

    Mirrors the emulated dumbbell's client/routerA namespaces with the far
    side replaced by the real path. The client uses a private address that
    root MASQUERADEs to the tailscale IP: the source stays 10.99.x through
    filter-FORWARD, so tailscale's anti-spoof rules (drop 100.64/10 from
    non-tailscale ifaces) never fire, and per-ns sysctls keep cc/buf changes
    off the host. Conntrack handles the return path."""
    ch("sudo ip netns del cl 2>/dev/null; sudo ip netns del mb 2>/dev/null; "
       "true")
    ch("sudo ip netns add cl; sudo ip netns add mb; "
       "sudo ip link add wca type veth peer name wcm; "
       "sudo ip link add wmh type veth peer name wmb; "
       "sudo ip link set wca netns cl; sudo ip link set wcm netns mb; "
       "sudo ip link set wmb netns mb; "
       "sudo ip addr add 10.99.2.2/24 dev wmh; "
       "sudo ip link set wmh up mtu 1280; "
       "sudo ip netns exec cl ip addr add 10.99.1.1/24 dev wca; "
       "sudo ip netns exec cl ip link set wca up mtu 1280; "
       "sudo ip netns exec cl ip link set lo up; "
       "sudo ip netns exec cl ip route add default via 10.99.1.2; "
       "sudo ip netns exec mb ip addr add 10.99.1.2/24 dev wcm; "
       "sudo ip netns exec mb ip link set wcm up mtu 1280; "
       "sudo ip netns exec mb ip addr add 10.99.2.1/24 dev wmb; "
       "sudo ip netns exec mb ip link set wmb up mtu 1280; "
       "sudo ip netns exec mb ip link set lo up; "
       "sudo ip netns exec mb ip route add default via 10.99.2.2")
    # packet-counted queues need MTU-sized packets, as in the testbed
    ch("sudo ip netns exec cl ethtool -K wca tso off gso off gro off; "
       "sudo ip netns exec mb ethtool -K wcm tso off gso off gro off; "
       "sudo ip netns exec mb ethtool -K wmb tso off gso off gro off; "
       "sudo ethtool -K wmh tso off gso off gro off")
    # veths created in the root ns inherit its conf.default.rp_filter (2 on
    # Ubuntu) and keep it across the netns move - zero them explicitly
    ch("sudo ip netns exec mb sysctl -qw net.ipv4.ip_forward=1 "
       "net.ipv4.conf.all.rp_filter=0 net.ipv4.conf.wcm.rp_filter=0 "
       "net.ipv4.conf.wmb.rp_filter=0; "
       "sudo sysctl -qw net.ipv4.ip_forward=1 "
       "net.ipv4.conf.wmh.rp_filter=0; "
       "sudo ip route replace 10.99.1.0/24 via 10.99.2.1 dev wmh; "
       "sudo iptables -t nat -C POSTROUTING -s 10.99.0.0/16 -o tailscale0 "
       "-j MASQUERADE 2>/dev/null || "
       "sudo iptables -t nat -A POSTROUTING -s 10.99.0.0/16 -o tailscale0 "
       "-j MASQUERADE; "
       "sudo iptables -C FORWARD -s 10.99.0.0/16 -j ACCEPT 2>/dev/null || "
       "sudo iptables -I FORWARD 1 -s 10.99.0.0/16 -j ACCEPT; "
       "sudo iptables -C FORWARD -d 10.99.0.0/16 -j ACCEPT 2>/dev/null || "
       "sudo iptables -I FORWARD 1 -d 10.99.0.0/16 -j ACCEPT")
    # firewalld's native nft forward hook rejects what iptables accepted;
    # runtime-only trust (cleared on reboot / teardown)
    ch("sudo firewall-cmd -q --zone=trusted --add-source=10.99.0.0/16; "
       f"sudo firewall-cmd -q --zone=trusted --add-source={LOCAL_TS}/32; true")


def teardown_middlebox() -> None:
    ch("sudo firewall-cmd -q --zone=trusted --remove-source=10.99.0.0/16 "
       "2>/dev/null; "
       f"sudo firewall-cmd -q --zone=trusted --remove-source={LOCAL_TS}/32 "
       "2>/dev/null; true")
    ch("sudo iptables -t nat -D POSTROUTING -s 10.99.0.0/16 -o tailscale0 "
       "-j MASQUERADE 2>/dev/null; "
       "sudo iptables -D FORWARD -s 10.99.0.0/16 -j ACCEPT 2>/dev/null; "
       "sudo iptables -D FORWARD -d 10.99.0.0/16 -j ACCEPT 2>/dev/null; "
       "sudo iptables -D FORWARD -o tailscale0 -s 100.73.56.42 -j ACCEPT 2>/dev/null; "
       "sudo ip route del 10.99.1.0/24 2>/dev/null; "
       "sudo ip rule del pref 40 2>/dev/null; "
       "sudo ip rule del pref 50 2>/dev/null; "
       "sudo ip route flush table 100 2>/dev/null; "
       "sudo ip route flush table 110 2>/dev/null; "
       f"sudo ip route del {LOCAL_TS}/32 2>/dev/null; "
       "sudo ip netns del cl 2>/dev/null; "
       "sudo ip netns del mb 2>/dev/null; true")


def apply_config(cfg: RunConfig) -> None:
    buf = BUF_PRESETS[cfg.buf]
    child = {
        "pfifo_deep": "pfifo limit 1000",
        "pfifo_bdp": f"pfifo limit {bdp_pkts(cfg.bw_mbit)}",
        "fq_codel": "fq_codel",
        "fq": "fq",
    }
    mb = "sudo ip netns exec mb"
    cl = "sudo ip netns exec cl"
    script = (
        f"{cl} sysctl -qw net.ipv4.tcp_congestion_control={cfg.cc}; "
        f"{cl} sysctl -qw net.ipv4.tcp_wmem='{buf['wmem']}'; "
        f"{cl} sysctl -qw net.ipv4.tcp_rmem='{buf['rmem']}'; "
        f"{mb} tc qdisc del dev wmb root 2>/dev/null; "
    )
    if cfg.qdisc == "cake":
        script += (f"{mb} tc qdisc add dev wmb root cake "
                   f"bandwidth {cfg.bw_mbit}mbit besteffort")
    else:
        script += (
            f"{mb} tc qdisc add dev wmb root handle 1: htb default 1; "
            f"{mb} tc class add dev wmb parent 1: classid 1:1 htb "
            f"rate {cfg.bw_mbit}mbit ceil {cfg.bw_mbit}mbit; "
            f"{mb} tc qdisc add dev wmb parent 1:1 {child[cfg.qdisc]}")
    ch(script)


def run_workload(wl: str, dur: int = 10) -> dict:
    if wl == "bulk":
        out = ch(f"sudo ip netns exec cl iperf3 -c {LOCAL_TS} -p 5201 -t {dur} -J", timeout=dur + 30)
        d = json.loads(out)
        return {"goodput_mbps": round(
            d["end"]["sum_received"]["bits_per_second"] / 1e6, 3),
            "retransmits": d["end"]["sum_sent"].get("retransmits", 0)}
    if wl == "mixed":
        out = ch(
            f"sudo ip netns exec cl iperf3 -c {LOCAL_TS} -p 5201 -t {dur + 2} -J "
            "> /tmp/bulk.json & "
            f"sleep 1; sudo ip netns exec cl python3 rr_tool.py rr-client "
            f"--host {LOCAL_TS} "
            f"--port 5288 --duration {dur}; wait; cat /tmp/bulk.json",
            timeout=dur + 40)
        rr_line, _, bulk_json = out.partition("\n{")
        rr = json.loads(rr_line)
        bulk = json.loads("{" + bulk_json)
        return {
            "rr_p50_ms": round(rr["lat_p50_ms"], 3),
            "rr_p99_ms": round(rr["lat_p99_ms"], 3),
            "goodput_mbps": round(
                bulk["end"]["sum_received"]["bits_per_second"] / 1e6, 3),
        }
    raise ValueError(wl)


def anchor_from(cfg: RunConfig, metrics: dict, metric: str) -> dict:
    d = cfg.describe()
    d["metrics"] = metrics
    d["primary_metric"] = metric
    d["primary_value"] = metrics.get(metric)
    return d


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # sender-side CC modules (lost on reboot, like the testbed)
    ch("sudo modprobe -a tcp_bbr tcp_vegas tcp_westwood tcp_illinois "
       "tcp_dctcp 2>/dev/null; true")
    setup_middlebox()

    traces = []
    try:
        for wl, bw in ENVS_W:
            metric, direction = PRIMARY[wl]
            cur = RunConfig("W", wl, "cubic", "pfifo_deep", RTT_MS, 0.0, bw,
                            "ample")
            visited = {knobs_of(cur.describe())}
            trace = {"env": f"{wl}_wan_b{bw}", "metric": metric,
                     "better": direction, "steps": []}
            print(f"\n== REAL PATH {trace['env']} ({metric}) ==", flush=True)
            for step in range(MAX_STEPS):
                apply_config(cur)
                time.sleep(1)
                metrics = run_workload(wl)
                val = metrics.get(metric)
                print(f"step {step}: {cur.cc}/{cur.qdisc}/{cur.buf} "
                      f"-> {metrics}", flush=True)
                anchor = anchor_from(cur, metrics, metric)
                nbrs = [(k, nb) for k, nb in neighbors(cur)
                        if knobs_of(nb.describe()) not in visited]
                preds = predict_neighbors(anchor, nbrs, metric,
                                          f"wan_{trace['env']}_s{step}")
                nxt, gain = gated_best(nbrs, preds, direction)
                trace["steps"].append({
                    "config": f"{cur.cc}/{cur.qdisc}/{cur.buf}",
                    "metrics": metrics,
                    "move": f"{nxt.cc}/{nxt.qdisc}/{nxt.buf}" if nxt else None,
                    "predicted_gain": round(gain, 3) if nxt else None,
                })
                if nxt is None:
                    print("  no confident improvement -> stop", flush=True)
                    break
                visited.add(knobs_of(nxt.describe()))
                cur = nxt
            else:
                # measure the final config if we exhausted steps on a move
                apply_config(cur)
                time.sleep(1)
                metrics = run_workload(wl)
                trace["steps"].append({
                    "config": f"{cur.cc}/{cur.qdisc}/{cur.buf}",
                    "metrics": metrics, "move": None, "predicted_gain": None})
            trace["final_config"] = f"{cur.cc}/{cur.qdisc}/{cur.buf}"
            trace["final"] = trace["steps"][-1]["metrics"]
            traces.append(trace)
            print(f"final: {trace['final_config']} -> {trace['final']}",
                  flush=True)
    finally:
        teardown_middlebox()
        ch("sudo sysctl -qw net.ipv4.tcp_congestion_control=cubic; true")

    (OUT_DIR / "realpath.json").write_text(json.dumps(traces, indent=1))
    print("\nREALPATH DONE", flush=True)


if __name__ == "__main__":
    main()
