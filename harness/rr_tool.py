"""Request-response and short-flow TCP workload tool.

Runs inside network namespaces via `ip netns exec`.  Modes:

  server            persistent server handling both protocols
  rr-client         fixed-size request/response pingpong on one connection,
                    reports per-transaction latency percentiles
  flow-client       N sequential connections, each fetching SIZE bytes,
                    reports per-flow completion times (FCT)

Protocol: client sends an 8-byte big-endian integer N; server replies with
N bytes.  rr uses N=64; flows use N=flow size.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time

RR_BYTES = 64


def serve(host: str, port: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(128)
    blob = b"x" * (1 << 20)

    def handle(conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                hdr = b""
                while len(hdr) < 8:
                    chunk = conn.recv(8 - len(hdr))
                    if not chunk:
                        return
                    hdr += chunk
                n = struct.unpack(">Q", hdr)[0]
                left = n
                while left > 0:
                    sent = conn.send(blob[: min(left, len(blob))])
                    left -= sent
        except OSError:
            pass
        finally:
            conn.close()

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def recv_exact(conn: socket.socket, n: int) -> None:
    left = n
    while left > 0:
        chunk = conn.recv(min(left, 1 << 16))
        if not chunk:
            raise ConnectionError("short read")
        left -= len(chunk)


def pctl(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
    return xs[k]


def rr_client(host: str, port: int, duration: float) -> dict:
    conn = socket.create_connection((host, port), timeout=30)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    req = struct.pack(">Q", RR_BYTES)
    lat_ms: list[float] = []
    end = time.monotonic() + duration
    while time.monotonic() < end:
        t0 = time.monotonic()
        conn.sendall(req)
        recv_exact(conn, RR_BYTES)
        lat_ms.append((time.monotonic() - t0) * 1e3)
    conn.close()
    return {
        "kind": "rr",
        "transactions": len(lat_ms),
        "rps": len(lat_ms) / duration,
        "lat_p50_ms": pctl(lat_ms, 50),
        "lat_p95_ms": pctl(lat_ms, 95),
        "lat_p99_ms": pctl(lat_ms, 99),
        "lat_mean_ms": sum(lat_ms) / max(1, len(lat_ms)),
        "lat_max_ms": max(lat_ms) if lat_ms else float("nan"),
    }


def flow_client(host: str, port: int, nflows: int, size: int) -> dict:
    fct_ms: list[float] = []
    for _ in range(nflows):
        t0 = time.monotonic()
        conn = socket.create_connection((host, port), timeout=60)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.sendall(struct.pack(">Q", size))
        recv_exact(conn, size)
        conn.close()
        fct_ms.append((time.monotonic() - t0) * 1e3)
    return {
        "kind": "flow",
        "flows": nflows,
        "flow_bytes": size,
        "fct_p50_ms": pctl(fct_ms, 50),
        "fct_p95_ms": pctl(fct_ms, 95),
        "fct_mean_ms": sum(fct_ms) / len(fct_ms),
        "fct_max_ms": max(fct_ms),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["server", "rr-client", "flow-client"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--flows", type=int, default=40)
    ap.add_argument("--size", type=int, default=200_000)
    args = ap.parse_args()
    if args.mode == "server":
        serve(args.host, args.port)
    elif args.mode == "rr-client":
        print(json.dumps(rr_client(args.host, args.port, args.duration)))
    else:
        print(json.dumps(flow_client(args.host, args.port, args.flows, args.size)))


if __name__ == "__main__":
    sys.exit(main())
