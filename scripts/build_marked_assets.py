#!/usr/bin/env python3
"""Build marker-token variants of the FT dataset and eval prompts.

Reads the canonical assets (dataset/ft-all/{train,val}.jsonl and
dataset/ft/eval_prompts.jsonl), wraps each user prompt with the marker
tokens from scripts/marks.py, and writes:

  dataset/ft-marked/{train,val}.jsonl
  dataset/ft/eval_prompts_marked.jsonl

Content is otherwise byte-identical, so any accuracy difference between a
marked and an unmarked model is attributable to the markers.

  python3 scripts/build_marked_assets.py
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from marks import add_marks, SPECIAL_TOKENS  # noqa: E402


def main() -> None:
    out_dir = REPO / "dataset" / "ft-marked"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        src = REPO / "dataset" / "ft-all" / f"{split}.jsonl"
        n_marked = 0
        with open(out_dir / f"{split}.jsonl", "w") as out:
            for line in open(src):
                r = json.loads(line)
                user = r["messages"][0]
                assert user["role"] == "user"
                marked = add_marks(user["content"])
                n_marked += marked != user["content"]
                user["content"] = marked
                out.write(json.dumps(r) + "\n")
        print(f"{split}: marked {n_marked} prompts -> ft-marked/{split}.jsonl")

    src = REPO / "dataset" / "ft" / "eval_prompts.jsonl"
    dst = REPO / "dataset" / "ft" / "eval_prompts_marked.jsonl"
    with open(dst, "w") as out:
        for line in open(src):
            r = json.loads(line)
            r["prompt"] = add_marks(r["prompt"])
            out.write(json.dumps(r) + "\n")
    print(f"eval prompts -> {dst.name}")
    print("special tokens:", " ".join(SPECIAL_TOKENS))


if __name__ == "__main__":
    main()
