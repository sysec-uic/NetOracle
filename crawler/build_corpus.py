"""Build a retrievable corpus from crawled kernel commits, netdev threads, and
in-tree networking documentation, with a self-contained BM25 index.

Doc types:
  doc     Documentation/networking/* chunks (sysctl semantics, CC algo docs)
  commit  net/ commit messages (mechanism rationale, Fixes: chains)
  thread  netdev patch-discussion threads (review debates, perf regressions)

Usage:
  python3 build_corpus.py --kernel data/kernel --netdev data/netdev \
      --linux-repo data/linux.git --out data/corpus
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pathlib
import pickle
import re
import subprocess
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-z0-9_]+")

DOC_FILES = [
    "Documentation/networking/ip-sysctl.rst",
    "Documentation/networking/scaling.rst",
    "Documentation/admin-guide/sysctl/net.rst",
]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, k1: float = 1.4, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        self.n_docs = 0
        self.avg_len = 0.0

    def add(self, tokens: list[str]) -> int:
        idx = self.n_docs
        tf = Counter(tokens)
        for t, c in tf.items():
            self.postings[t].append((idx, c))
        self.doc_len.append(len(tokens))
        self.n_docs += 1
        return idx

    def finalize(self) -> None:
        self.avg_len = sum(self.doc_len) / max(1, self.n_docs)
        self.postings = dict(self.postings)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(tokenize(query)):
            plist = self.postings.get(t)
            if not plist:
                continue
            idf = math.log(1 + (self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for idx, tf in plist:
                dl = self.doc_len[idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                scores[idx] += idf * tf * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]


def chunk_doc(text: str, title: str, max_lines: int = 80) -> list[tuple[str, str]]:
    lines = text.splitlines()
    out = []
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i : i + max_lines]).strip()
        if chunk:
            out.append((f"{title}#{i // max_lines}", chunk))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--netdev", required=True)
    ap.add_argument("--linux-repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-threads", type=int, default=150_000)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    docs: list[dict] = []

    # 1. in-tree documentation (fetched lazily from the blobless clone)
    for path in DOC_FILES:
        r = subprocess.run(["git", "-C", args.linux_repo, "show", f"HEAD:{path}"],
                           capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            print(f"skip {path}: not found")
            continue
        for title, chunk in chunk_doc(r.stdout, path):
            docs.append({"type": "doc", "title": title, "text": chunk})
    n_doc = len(docs)

    # 2. kernel commits
    with gzip.open(pathlib.Path(args.kernel) / "commits.jsonl.gz", "rt") as f:
        for line in f:
            c = json.loads(line)
            docs.append({
                "type": "commit",
                "title": c["subject"],
                "text": (c["subject"] + "\n" + c["body"])[:6000],
                "sha": c["sha"][:12],
                "is_fix": c["is_fix"],
            })
    n_commit = len(docs) - n_doc

    # 3. netdev discussion threads (need bodies -> join messages onto threads)
    ndv = pathlib.Path(args.netdev)
    bodies: dict[str, str] = {}
    with gzip.open(ndv / "messages.jsonl.gz", "rt") as f:
        for line in f:
            m = json.loads(line)
            # keep discussion prose, drop the diff itself
            body = m["body"].split("\ndiff --git ")[0]
            bodies[m["mid"]] = body[:4000]
    n_threads = 0
    with gzip.open(ndv / "threads.jsonl.gz", "rt") as f:
        for line in f:
            t = json.loads(line)
            if not t["is_patch_thread"] or t["n_messages"] < 2:
                continue
            if n_threads >= args.max_threads:
                break
            text = "\n---\n".join(bodies.get(mid, "") for mid in t["mids"][:8])
            docs.append({
                "type": "thread",
                "title": t["subject"],
                "text": (t["subject"] + "\n" + text)[:8000],
                "n_messages": t["n_messages"],
            })
            n_threads += 1

    print(f"docs: {n_doc} doc-chunks, {n_commit} commits, {n_threads} threads")

    bm25 = BM25()
    for d in docs:
        bm25.add(tokenize(d["title"] + " " + d["text"]))
    bm25.finalize()

    with gzip.open(out / "docs.jsonl.gz", "wt") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    with open(out / "bm25.pkl", "wb") as f:
        pickle.dump({"postings": bm25.postings, "doc_len": bm25.doc_len,
                     "n_docs": bm25.n_docs, "avg_len": bm25.avg_len}, f)
    stats = {"n_doc_chunks": n_doc, "n_commits": n_commit, "n_threads": n_threads,
             "n_total": len(docs)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


class Retriever:
    """Load a built corpus and answer queries (used by eval/)."""

    def __init__(self, corpus_dir: str | pathlib.Path) -> None:
        corpus_dir = pathlib.Path(corpus_dir)
        with open(corpus_dir / "bm25.pkl", "rb") as f:
            state = pickle.load(f)
        self.bm25 = BM25()
        self.bm25.postings = state["postings"]
        self.bm25.doc_len = state["doc_len"]
        self.bm25.n_docs = state["n_docs"]
        self.bm25.avg_len = state["avg_len"]
        self.docs: list[dict] = []
        with gzip.open(corpus_dir / "docs.jsonl.gz", "rt") as f:
            for line in f:
                self.docs.append(json.loads(line))

    def search(self, query: str, k: int = 5, types: tuple = ("doc", "commit", "thread")) -> list[dict]:
        hits = []
        seen_titles = set()
        for idx, score in self.bm25.search(query, k * 6):
            d = self.docs[idx]
            if d["type"] not in types:
                continue
            # collapse patch-series revisions ([PATCH v2 ...] resends etc.)
            norm = re.sub(r"\[[^\]]*\]", "", d["title"]).strip().lower()
            if norm in seen_titles:
                continue
            seen_titles.add(norm)
            hits.append({**d, "score": round(score, 2)})
            if len(hits) >= k:
                break
        return hits


if __name__ == "__main__":
    main()
