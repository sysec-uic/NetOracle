#!/usr/bin/env python3
"""Marked vs unmarked comparison across model sizes (headline columns on
the 218 leakage-clean tasks).

  python3 scripts/marks_report.py
"""

from __future__ import annotations

import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_stats import headline, load  # noqa: E402

PAIRS = [
    ("0.8B unmarked", ["ft-r2-08b-s41", "ft-r2-08b-s42", "ft-r2-08b-s43"]),
    ("0.8B marked",   ["ft-mk-08b-s41", "ft-mk-08b-s42", "ft-mk-08b-s43"]),
    ("2B unmarked",   ["ft-r2-2b-s41", "ft-r2-2b-s42", "ft-r2-2b-s43"]),
    ("2B marked",     ["ft-mk-2b-s41", "ft-mk-2b-s42", "ft-mk-2b-s43"]),
    ("9B unmarked",   ["ft-r2-9b-s41"]),
    ("9B marked",     ["ft-mk-9b-s41"]),
    ("Gemma4-12B unmarked", ["ft-r2-gemma12b-s41"]),
    ("Gemma4-12B marked",   ["ft-mk-gemma12b-s41"]),
]
KEYS = ("dir_all", "dir_offgrid", "med_ape", "w2x_offgrid")


def main() -> None:
    print(f"{'group':22s} dir_all      dir_off      medAPE       w2x_off")
    for name, seeds in PAIRS:
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
                cells.append(
                    f"{statistics.mean(vals):.3f}±{statistics.stdev(vals):.3f}")
        print(f"{name:22s} " + "  ".join(f"{c:11s}" for c in cells)
              + f"  ({len(hs)} seeds)")


if __name__ == "__main__":
    main()
