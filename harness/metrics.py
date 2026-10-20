"""Parsers for kernel-side observables: ss -ti, /proc/net counters, tc -s."""

from __future__ import annotations

import re


def parse_ss_tcp_info(text: str) -> dict:
    """Parse the first established-connection block of `ss -tin` output."""
    out: dict = {}
    m = re.search(r"\bcwnd:(\d+)", text)
    if m:
        out["cwnd"] = int(m.group(1))
    m = re.search(r"\brtt:([\d.]+)/([\d.]+)", text)
    if m:
        out["srtt_ms"] = float(m.group(1))
        out["rttvar_ms"] = float(m.group(2))
    m = re.search(r"\bretrans:\d+/(\d+)", text)
    if m:
        out["retrans_total"] = int(m.group(1))
    m = re.search(r"\bdelivery_rate\s+([\d.]+)([MKG]?)bps", text)
    if m:
        mult = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]
        out["delivery_rate_mbps"] = float(m.group(1)) * mult / 1e6
    m = re.search(r"\bpacing_rate\s+([\d.]+)([MKG]?)bps", text)
    if m:
        mult = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]
        out["pacing_rate_mbps"] = float(m.group(1)) * mult / 1e6
    m = re.search(r"\bssthresh:(\d+)", text)
    if m:
        out["ssthresh"] = int(m.group(1))
    m = re.search(r"\bbytes_retrans:(\d+)", text)
    if m:
        out["bytes_retrans"] = int(m.group(1))
    return out


def parse_ss_best(text: str) -> dict:
    """Split `ss -tin` output into per-connection blocks and return the data
    connection (max bytes_acked, falling back to max cwnd) — iperf3 keeps a
    mostly-idle control connection on the same port."""
    blocks: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        if line.startswith(("ESTAB", "SYN", "FIN", "CLOSE", "LAST", "TIME")):
            if cur:
                blocks.append("\n".join(cur))
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    best: dict = {}
    best_key = (-1, -1)
    for b in blocks:
        d = parse_ss_tcp_info(b)
        if not d:
            continue
        m = re.search(r"\bbytes_acked:(\d+)", b)
        key = (int(m.group(1)) if m else 0, d.get("cwnd", 0))
        if key > best_key:
            best_key = key
            best = d
    return best


def parse_proc_counters(text: str) -> dict[str, int]:
    """Parse /proc/net/snmp + /proc/net/netstat header/value line pairs."""
    out: dict[str, int] = {}
    lines = text.splitlines()
    for i in range(0, len(lines) - 1):
        h, v = lines[i], lines[i + 1]
        if ":" not in h or ":" not in v:
            continue
        hp, hf = h.split(":", 1)
        vp, vf = v.split(":", 1)
        if hp != vp or hf == vf:
            continue
        keys, vals = hf.split(), vf.split()
        if len(keys) != len(vals) or not vals or not vals[0].lstrip("-").isdigit():
            continue
        for k, x in zip(keys, vals):
            out[f"{hp}.{k}"] = int(x)
    return out


TRACKED_COUNTERS = [
    "Tcp.RetransSegs",
    "Tcp.OutSegs",
    "TcpExt.TCPTimeouts",
    "TcpExt.TCPFastRetrans",
    "TcpExt.TCPSlowStartRetrans",
    "TcpExt.TCPLostRetransmit",
    "TcpExt.TCPSACKReorder",
    "TcpExt.TCPDSACKRecv",
]


def counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        k.replace(".", "_"): after.get(k, 0) - before.get(k, 0)
        for k in TRACKED_COUNTERS
        if k in after or k in before
    }


def parse_qdisc_stats(text: str) -> dict:
    """Aggregate sent/dropped/overlimits/max-backlog across the qdisc tree."""
    out = {"qdisc_sent_bytes": 0, "qdisc_dropped": 0, "qdisc_overlimits": 0}
    for m in re.finditer(
        r"Sent (\d+) bytes \d+ pkt \(dropped (\d+), overlimits (\d+)", text
    ):
        out["qdisc_sent_bytes"] = max(out["qdisc_sent_bytes"], int(m.group(1)))
        out["qdisc_dropped"] += int(m.group(2))
        out["qdisc_overlimits"] += int(m.group(3))
    backlogs = [int(b) for b in re.findall(r"backlog (\d+)b", text)]
    if backlogs:
        out["qdisc_backlog_bytes"] = max(backlogs)
    return out


def qdisc_delta(before: dict, after: dict) -> dict:
    return {
        k: after.get(k, 0) - before.get(k, 0)
        for k in ("qdisc_sent_bytes", "qdisc_dropped", "qdisc_overlimits")
    } | {"qdisc_backlog_bytes": after.get("qdisc_backlog_bytes", 0)}
