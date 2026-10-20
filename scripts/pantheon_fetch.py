#!/usr/bin/env python3
"""Fetch Pantheon (Stanford) real-world measurement summaries.

Enumerates the public stanford-pantheon S3 bucket, downloads every
*-pantheon-perf.json under real-world/<node>/reports/, and stores them in
dataset/pantheon/raw/. These are per-experiment summaries: for each scheme
and run, mean throughput (Mbit/s), 95th-percentile one-way delay (ms), and
loss rate, on real Internet paths (wired, cellular, inter-datacenter).

  python3 scripts/pantheon_fetch.py [--max-per-node 40]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import urllib.request
import xml.etree.ElementTree as ET

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "dataset" / "pantheon" / "raw"
BUCKET = "https://s3.amazonaws.com/stanford-pantheon"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_keys(prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            tree = ET.fromstring(r.read())
        keys += [c.find(NS + "Key").text for c in tree.iter(NS + "Contents")]
        if tree.find(NS + "IsTruncated").text != "true":
            return keys
        token = tree.find(NS + "NextContinuationToken").text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-node", type=int, default=40)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import urllib.parse  # noqa: F401 (used in list_keys via urllib)

    nodes = []
    with urllib.request.urlopen(
            f"{BUCKET}/?list-type=2&prefix=real-world/&delimiter=/", timeout=60) as r:
        tree = ET.fromstring(r.read())
    for p in tree.iter(NS + "CommonPrefixes"):
        nodes.append(p.find(NS + "Prefix").text)
    print(f"{len(nodes)} nodes")

    n_dl = 0
    for node in nodes:
        keys = [k for k in list_keys(node + "reports/")
                if k.endswith("pantheon-perf.json")]
        # spread across time: take evenly spaced sample
        if len(keys) > args.max_per_node:
            step = len(keys) / args.max_per_node
            keys = [keys[int(i * step)] for i in range(args.max_per_node)]
        print(f"{node}: {len(keys)} perf files")
        for k in keys:
            fn = OUT / k.replace("real-world/", "").replace("/", "__")
            if fn.exists():
                continue
            try:
                with urllib.request.urlopen(f"{BUCKET}/{k}", timeout=120) as r:
                    data = r.read()
                json.loads(data)  # validate
                fn.write_bytes(data)
                n_dl += 1
            except Exception as e:
                print(f"  skip {k}: {e}")
    print(f"downloaded {n_dl} new files; total {len(list(OUT.glob('*.json')))}")


if __name__ == "__main__":
    import urllib.parse
    main()
