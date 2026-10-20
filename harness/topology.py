"""Network-namespace dumbbell testbed.

Topology (all veth, all emulation off the end hosts):

  nm_cl (client)          nm_ra (routerA)            nm_rb (routerB)         nm_sv (server)
  10.0.1.1  vca <----> vac 10.0.1.2                                          10.0.3.2
                           10.0.2.1 vab <----> vba 10.0.2.2
                                                    10.0.3.1 vbs <----> vsb

  * routerA egress vab   : bottleneck (rate shaping + AQM qdisc), data direction
  * routerB egress vbs   : netem delay rtt/2 + loss   (data direction)
  * routerB egress vba   : netem delay rtt/2          (ack direction)

Data flows client -> server.  All privileged operations go through sudo so the
runner itself stays an unprivileged process and owns its output files.
"""

from __future__ import annotations

import subprocess
import time

from config_space import BUF_PRESETS, RunConfig

NS = {"cl": "nm_cl", "ra": "nm_ra", "rb": "nm_rb", "sv": "nm_sv"}
CLIENT_IP = "10.0.1.1"
SERVER_IP = "10.0.3.2"
NETEM_LIMIT = 100000  # packets; large enough to never be the implicit bottleneck


def sh(cmd: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0 and not quiet:
        raise RuntimeError(f"cmd failed ({r.returncode}): {cmd}\n{r.stderr.strip()}")
    return r


def ns_exec(ns: str, cmd: str) -> str:
    return f"sudo ip netns exec {NS[ns]} {cmd}"


def destroy() -> None:
    for ns in NS.values():
        sh(f"sudo ip netns del {ns}", check=False)


def create() -> None:
    destroy()
    # CC modules are not autoloaded after a reboot; without them the allowed
    # list below silently drops the missing algorithms.
    sh("sudo modprobe -a tcp_bbr tcp_vegas tcp_westwood tcp_illinois tcp_dctcp",
       check=False)
    # Non-init netns may only set CC algos listed in the init-ns allowed list.
    sh("sudo sysctl -qw net.ipv4.tcp_allowed_congestion_control="
       "'reno cubic bbr vegas westwood illinois dctcp'", check=False)
    for ns in NS.values():
        sh(f"sudo ip netns add {ns}")
    links = [
        ("vca", "cl", "10.0.1.1/24", "vac", "ra", "10.0.1.2/24"),
        ("vab", "ra", "10.0.2.1/24", "vba", "rb", "10.0.2.2/24"),
        ("vbs", "rb", "10.0.3.1/24", "vsb", "sv", "10.0.3.2/24"),
    ]
    for a, ans, aip, b, bns, bip in links:
        sh(f"sudo ip link add {a} type veth peer name {b}")
        sh(f"sudo ip link set {a} netns {NS[ans]}")
        sh(f"sudo ip link set {b} netns {NS[bns]}")
        for dev, ns, ip in [(a, ans, aip), (b, bns, bip)]:
            sh(ns_exec(ns, f"ip addr add {ip} dev {dev}"))
            sh(ns_exec(ns, f"ip link set {dev} up"))
            sh(ns_exec(ns, f"ip link set lo up"))
            # Segmentation offloads make netem loss/rate act on 64KB
            # super-packets; disable for faithful per-packet emulation.
            sh(ns_exec(ns, f"ethtool -K {dev} tso off gso off gro off"), check=False)
    # routing
    sh(ns_exec("cl", "ip route add default via 10.0.1.2"))
    sh(ns_exec("sv", "ip route add default via 10.0.3.1"))
    sh(ns_exec("ra", "ip route add 10.0.3.0/24 via 10.0.2.2"))
    sh(ns_exec("rb", "ip route add 10.0.1.0/24 via 10.0.2.1"))
    for r in ("ra", "rb"):
        sh(ns_exec(r, "sysctl -qw net.ipv4.ip_forward=1"))
    # end-host hygiene: no cached metrics leaking across runs
    for h in ("cl", "sv"):
        sh(ns_exec(h, "sysctl -qw net.ipv4.tcp_no_metrics_save=1"))
    ping = sh(ns_exec("cl", f"ping -c 1 -W 2 {SERVER_IP}"), check=False)
    if ping.returncode != 0:
        raise RuntimeError(f"topology self-test ping failed:\n{ping.stdout}{ping.stderr}")


def _clear_qdiscs() -> None:
    for ns, dev in [("ra", "vab"), ("ra", "vac"), ("rb", "vbs"), ("rb", "vba")]:
        sh(ns_exec(ns, f"tc qdisc del dev {dev} root"), check=False)


def apply(cfg: RunConfig) -> None:
    """Reconfigure the standing topology for one run."""
    _clear_qdiscs()
    half = cfg.rtt_ms / 2
    loss = f" loss {cfg.loss_pct}%" if cfg.loss_pct > 0 else ""
    # path emulation on routerB
    sh(ns_exec("rb", f"tc qdisc add dev vbs root netem delay {half}ms{loss} limit {NETEM_LIMIT}"))
    sh(ns_exec("rb", f"tc qdisc add dev vba root netem delay {half}ms limit {NETEM_LIMIT}"))
    # bottleneck on routerA egress toward server
    if cfg.qdisc == "cake":
        sh(ns_exec("ra", f"tc qdisc add dev vab root cake bandwidth {cfg.bw_mbit}mbit besteffort"))
    else:
        sh(ns_exec("ra", "tc qdisc add dev vab root handle 1: htb default 1"))
        sh(ns_exec("ra", f"tc class add dev vab parent 1: classid 1:1 htb rate {cfg.bw_mbit}mbit ceil {cfg.bw_mbit}mbit"))
        child = {
            "pfifo_deep": "pfifo limit 1000",
            "pfifo_bdp": f"pfifo limit {cfg.bdp_pkts}",
            "fq_codel": "fq_codel",
            "fq": "fq",
        }[cfg.qdisc]
        sh(ns_exec("ra", f"tc qdisc add dev vab parent 1:1 handle 10: {child}"))
    # end-host stack config
    bufs = BUF_PRESETS[cfg.buf]
    for h in ("cl", "sv"):
        sh(ns_exec(h, f"sysctl -qw net.ipv4.tcp_congestion_control={cfg.cc}"))
        sh(ns_exec(h, f"sysctl -qw net.ipv4.tcp_wmem='{bufs['wmem']}'"))
        sh(ns_exec(h, f"sysctl -qw net.ipv4.tcp_rmem='{bufs['rmem']}'"))
    time.sleep(0.1)


def qdisc_stats(ns: str = "ra", dev: str = "vab") -> str:
    return sh(ns_exec(ns, f"tc -s qdisc show dev {dev}")).stdout


def ss_tcp_info(ns: str = "cl", dst: str = SERVER_IP) -> str:
    return sh(ns_exec(ns, f"ss -tin dst {dst}")).stdout


def proc_netstat(ns: str) -> str:
    a = sh(ns_exec(ns, "cat /proc/net/snmp")).stdout
    b = sh(ns_exec(ns, "cat /proc/net/netstat")).stdout
    return a + b


if __name__ == "__main__":
    from config_space import smoke_matrix

    create()
    print("topology up, ping OK")
    apply(smoke_matrix()[0])
    print(qdisc_stats())
    rtt = sh(ns_exec("cl", f"ping -c 3 -i 0.2 {SERVER_IP}")).stdout
    print(rtt.splitlines()[-1])
    destroy()
    print("destroyed")
