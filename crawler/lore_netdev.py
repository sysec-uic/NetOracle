"""Parse a lore.kernel.org public-inbox git epoch into message + thread JSONL.

public-inbox stores one message per commit as the file `m`.  We stream all
(commit -> blob) pairs and batch-read the blobs, then reassemble threads via
Message-ID / References headers.

Usage:
  python3 lore_netdev.py --repo data/netdev-3.git --out data/netdev
"""

from __future__ import annotations

import argparse
import email
import email.policy
import email.utils
import gzip
import json
import pathlib
import re
import subprocess
from collections import defaultdict

PATCH_RE = re.compile(r"\[(?:RFC )?PATCH[^\]]*\]", re.I)


def iter_message_blobs(repo: str):
    """Yield raw message bytes for every message in the epoch."""
    ls = subprocess.Popen(
        ["git", "-C", repo, "log", "--format=%H", "--raw", "--no-renames", "--all"],
        stdout=subprocess.PIPE,
        text=True,
    )
    blob_shas = []
    for line in ls.stdout:
        # raw line: :000000 100644 0000000 <blobsha> A\tm
        if line.startswith(":") and line.rstrip().endswith("\tm"):
            parts = line.split()
            if len(parts) >= 4 and parts[4] in ("A", "M"):
                blob_shas.append(parts[3])
    ls.wait()
    cat = subprocess.Popen(
        ["git", "-C", repo, "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    for sha in blob_shas:
        cat.stdin.write((sha + "\n").encode())
        cat.stdin.flush()
        hdr = cat.stdout.readline().decode()
        if "missing" in hdr:
            continue
        size = int(hdr.split()[2])
        data = cat.stdout.read(size)
        cat.stdout.read(1)  # trailing newline
        yield data
    cat.stdin.close()
    cat.wait()


def parse_message(raw: bytes) -> dict | None:
    try:
        msg = email.message_from_bytes(raw, policy=email.policy.default)
    except Exception:
        return None
    mid = (msg.get("Message-ID") or "").strip().strip("<>")
    if not mid:
        return None
    refs = re.findall(r"<([^>]+)>", (msg.get("References") or "") + " " + (msg.get("In-Reply-To") or ""))
    body = ""
    try:
        part = msg.get_body(preferencelist=("plain",))
        if part is not None:
            body = part.get_content()
    except Exception:
        pass
    subject = str(msg.get("Subject") or "").replace("\n", " ").strip()
    date = msg.get("Date")
    try:
        ts = email.utils.parsedate_to_datetime(date).timestamp() if date else None
    except Exception:
        ts = None
    has_diff = "\ndiff --git " in body or body.startswith("diff --git ")
    return {
        "mid": mid,
        "subject": subject,
        "from": str(msg.get("From") or ""),
        "ts": ts,
        "refs": refs,
        "is_patch": bool(PATCH_RE.search(subject)) or has_diff,
        "has_diff": has_diff,
        "body": body[:20000],
    }


def normalize_subject(s: str) -> str:
    s = re.sub(r"\[[^\]]*\]", "", s)  # strip [PATCH net-next v3 1/4] etc.
    s = re.sub(r"^(re|fwd):\s*", "", s.strip(), flags=re.I)
    return re.sub(r"\s+", " ", s).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    msgs: dict[str, dict] = {}
    n_raw = 0
    with gzip.open(out / "messages.jsonl.gz", "wt") as f:
        for raw in iter_message_blobs(args.repo):
            n_raw += 1
            if args.limit and n_raw > args.limit:
                break
            m = parse_message(raw)
            if m is None:
                continue
            msgs[m["mid"]] = {k: m[k] for k in ("mid", "subject", "ts", "refs", "is_patch")}
            f.write(json.dumps(m) + "\n")
            if n_raw % 20000 == 0:
                print(f"parsed {n_raw} messages", flush=True)

    # thread assembly: root = transitive first reference that exists in corpus
    root_of: dict[str, str] = {}

    def find_root(mid: str, depth: int = 0) -> str:
        if mid in root_of:
            return root_of[mid]
        m = msgs.get(mid)
        root = mid
        if m and m["refs"] and depth < 50:
            first = m["refs"][0]
            root = find_root(first, depth + 1) if first in msgs else first
        root_of[mid] = root
        return root

    threads: dict[str, list[str]] = defaultdict(list)
    for mid in msgs:
        threads[find_root(mid)].append(mid)

    with gzip.open(out / "threads.jsonl.gz", "wt") as f:
        for root, mids in threads.items():
            mids.sort(key=lambda m: msgs[m]["ts"] or 0)
            rootmsg = msgs.get(root) or msgs[mids[0]]
            f.write(json.dumps({
                "root": root,
                "subject": rootmsg["subject"],
                "norm_subject": normalize_subject(rootmsg["subject"]),
                "n_messages": len(mids),
                "is_patch_thread": any(msgs[m]["is_patch"] for m in mids),
                "mids": mids,
            }) + "\n")
    print(f"messages={len(msgs)} threads={len(threads)} "
          f"patch_threads={sum(1 for t in threads.values() if any(msgs[m]['is_patch'] for m in t))}")


if __name__ == "__main__":
    main()
