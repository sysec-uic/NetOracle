"""Extract Linux kernel network-stack commit history into JSONL.

Pulls commits touching the core network stack (net/, include/net/) from a
(blobless, shallow) kernel clone, with subject/message/Fixes: tags and
changed-file stats.

Usage:
  python3 kernel_git.py --repo data/linux.git --out data/kernel
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import re
import subprocess

SEP = "\x1e"  # record separator
FIELD = "\x1f"
FIXES_RE = re.compile(r"^Fixes:\s*([0-9a-f]{8,40})\b", re.M | re.I)
LINK_RE = re.compile(r"^Link:\s*(\S+)", re.M | re.I)

PATHS = ["net/", "include/net/", "include/linux/tcp.h", "include/linux/skbuff.h",
         "include/uapi/linux/tcp.h", "Documentation/networking/"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # SEP leads each record so the --name-only file list stays attached to
    # its own commit (git prints it after the formatted fields).
    fmt = SEP + FIELD.join(["%H", "%an", "%at", "%s", "%b"])
    cmd = ["git", "-C", args.repo, "log", "--no-merges", f"--format={fmt}",
           "--name-only", "--"] + PATHS
    raw = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if raw.returncode != 0:
        raise SystemExit(f"git log failed: {raw.stderr[:500]}")

    path_re = re.compile(r"^[\w./+-]+$")
    n = 0
    with gzip.open(out / "commits.jsonl.gz", "wt") as f:
        for rec in raw.stdout.split(SEP):
            parts = rec.split(FIELD)
            if len(parts) < 5:
                continue
            sha, author, ts, subject = parts[0].strip(), parts[1], parts[2], parts[3]
            tail = parts[4].splitlines()
            # trailing block of path-looking lines = the --name-only list
            files: list[str] = []
            while tail and (not tail[-1].strip() or
                            (path_re.match(tail[-1]) and "/" in tail[-1])):
                l = tail.pop().strip()
                if l:
                    files.append(l)
            files.reverse()
            body = "\n".join(tail).strip()
            f.write(json.dumps({
                "sha": sha,
                "author": author,
                "ts": int(ts) if ts.strip().isdigit() else None,
                "subject": subject,
                "body": body[:8000],
                "fixes": [m.lower()[:12] for m in FIXES_RE.findall(body)],
                "links": LINK_RE.findall(body),
                "files": files[:50],
                "is_fix": bool(FIXES_RE.search(body)) or subject.lower().startswith("fix"),
            }) + "\n")
            n += 1
            if n % 10000 == 0:
                print(f"{n} commits", flush=True)
    print(f"total commits: {n}")


if __name__ == "__main__":
    main()
