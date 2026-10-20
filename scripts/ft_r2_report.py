#!/usr/bin/env python3
"""Headline-column report for the round-2 FT scale comparison.

  python3 scripts/ft_r2_report.py
"""

from __future__ import annotations

import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_stats import EVAL_DIR, headline, load  # noqa: E402

GROUPS = {
    "ft-all 0.8B (paper)": ["ft-all-r0-s41", "ft-all-r0-s42", "ft-all-r0-s43"],
    "ft-r2 0.8B (reproduced)": ["ft-r2-08b-s41", "ft-r2-08b-s42", "ft-r2-08b-s43"],
    "ft-r2 2B": ["ft-r2-2b-s41", "ft-r2-2b-s42", "ft-r2-2b-s43"],
    "ft-r2 9B": ["ft-r2-9b-s41"],
    "ft-v2 0.8B (more data)": ["ft-v2-08b-s41", "ft-v2-08b-s42", "ft-v2-08b-s43"],
}
KEYS = ("dir_all", "dir_offgrid", "med_ape", "w2x_offgrid")


def main() -> None:
    print(f"{'group':26s} dir_all      dir_off      medAPE       w2x_off")
    for name, seeds in GROUPS.items():
        hs = []
        for s in seeds:
            try:
                hs.append(headline(load(f"{s}|L0")))
            except FileNotFoundError:
                pass
        if not hs:
            continue
        cells = []
        for k in KEYS:
            vals = [h[k] for h in hs if h[k] is not None]
            if not vals:
                cells.append("--")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.3f}")
            else:
                cells.append(f"{statistics.mean(vals):.3f}±{statistics.stdev(vals):.3f}")
        print(f"{name:26s} " + "  ".join(f"{c:11s}" for c in cells)
              + f"  ({len(hs)} seeds)")


if __name__ == "__main__":
    main()
