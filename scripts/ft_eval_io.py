#!/usr/bin/env python3
"""Export FT-clean eval prompts / score generations back into results/eval.

  export: write dataset/ft/eval_prompts.jsonl  {task_id, type, split, knob, prompt}
  score:  read a generations file {task_id, text}, score with the standard
          scorers, write results/eval/<cond>.jsonl compatible with ft_compare

  python3 scripts/ft_eval_io.py export
  python3 scripts/ft_eval_io.py score --generations gen.jsonl --cond "ft-qwen3.5-2b|L0"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "eval"))

from build_ft_dataset import committed_cells  # noqa: E402
from llm import extract_json  # noqa: E402
from metrics import SCORERS  # noqa: E402
from prompts import BUILDERS  # noqa: E402
from tasks import build_all  # noqa: E402

FT_DIR = REPO / "dataset" / "ft"
EVAL_DIR = REPO / "results" / "eval"


def indexed_tasks() -> dict[str, dict]:
    cells = committed_cells()
    tasks = build_all(cells)
    return {f"{t}:{i}": task for t, lst in tasks.items()
            for i, task in enumerate(lst)}


def do_export(level: str) -> None:
    ids = json.loads((FT_DIR / "manifest.json").read_text())["ft_eval_task_ids"]
    tasks = indexed_tasks()
    cells = committed_cells()
    out = FT_DIR / "eval_prompts.jsonl"
    with open(out, "w") as f:
        for tid in ids:
            task = tasks[tid]
            prompt = BUILDERS[task["type"]](task, level, cells)
            f.write(json.dumps({
                "task_id": tid, "type": task["type"], "split": task["split"],
                "knob": task.get("knob") or task.get("vary"),
                "prompt": prompt,
            }) + "\n")
    print(f"{len(ids)} prompts (level {level}) -> {out}")


def do_score(gen_path: pathlib.Path, cond: str) -> None:
    tasks = indexed_tasks()
    out = EVAL_DIR / f"{cond}.jsonl"
    n = 0
    with open(out, "w") as f:
        for line in gen_path.read_text().splitlines():
            g = json.loads(line)
            task = tasks[g["task_id"]]
            answer = extract_json(g["text"])
            scores = SCORERS[task["type"]](task, answer)
            f.write(json.dumps({
                "task_id": g["task_id"], "type": task["type"],
                "split": task["split"],
                "knob": task.get("knob") or task.get("vary"),
                "answer": answer, "scores": scores,
            }) + "\n")
            n += 1
    print(f"{n} rows -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--level", default="L0")
    s = sub.add_parser("score")
    s.add_argument("--generations", required=True, type=pathlib.Path)
    s.add_argument("--cond", required=True)
    args = ap.parse_args()
    if args.cmd == "export":
        do_export(args.level)
    else:
        do_score(args.generations, args.cond)


if __name__ == "__main__":
    main()
