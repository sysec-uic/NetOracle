#!/usr/bin/env python3
"""Marker-token wrapping for world-model prompts (PF-LLM <load> analog).

PF-LLM adds <load></load> special tokens to the embedding table so the
model can locate the subject of the question inside a long context. Our
analogs:

  <anchor>...</anchor>  the measured anchor run (counterfactual tasks)
  <target>...</target>  the subject of the question: the one-knob change
                        sentence (counterfactual) or the target
                        configuration line (quantitative)

The wrapper is a pure string transform over the canonical prompts, so
marked datasets/eval prompts are guaranteed identical to the unmarked
ones except for the four marker tokens. Fails loudly if an expected
anchor string is missing, so format drift cannot silently produce
unmarked training data.
"""

from __future__ import annotations

SPECIAL_TOKENS = ["<anchor>", "</anchor>", "<target>", "</target>"]

ANCHOR_HEAD = "A measured run on this testbed:\n"
CF_TARGET_HEAD = "Now exactly one knob changes:"
CF_TARGET_TAIL = "Everything else stays the same."
Q_TARGET_HEAD = "Target configuration (never executed for you):\n"


def add_marks(prompt: str) -> str:
    out = prompt
    if ANCHOR_HEAD in out:  # counterfactual
        head = out.index(ANCHOR_HEAD) + len(ANCHOR_HEAD)
        line_end = out.index("\n\n", head)
        out = (out[:head] + "<anchor>" + out[head:line_end] + "</anchor>"
               + out[line_end:])
        assert CF_TARGET_HEAD in out and CF_TARGET_TAIL in out, "cf format drift"
        s = out.index(CF_TARGET_HEAD)
        e = out.index(CF_TARGET_TAIL) + len(CF_TARGET_TAIL)
        out = out[:s] + "<target>" + out[s:e] + "</target>" + out[e:]
        return out
    if Q_TARGET_HEAD in out:  # quantitative
        head = out.index(Q_TARGET_HEAD) + len(Q_TARGET_HEAD)
        line_end = out.index("\n\n", head)
        out = (out[:head] + "<target>" + out[head:line_end] + "</target>"
               + out[line_end:])
        return out
    return prompt  # ranking prompts unchanged


if __name__ == "__main__":
    import json
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "eval"))
    # self-test on the real assets
    n_cf = n_q = n_plain = 0
    for line in open("dataset/ft/eval_prompts.jsonl"):
        r = json.loads(line)
        m = add_marks(r["prompt"])
        if "<anchor>" in m:
            n_cf += 1
        elif "<target>" in m:
            n_q += 1
        else:
            n_plain += 1
        stripped = m
        for t in SPECIAL_TOKENS:
            stripped = stripped.replace(t, "")
        assert stripped == r["prompt"], "marks are not a pure wrap"
    print(f"self-test ok: cf-marked={n_cf} q-marked={n_q} unmarked={n_plain}")
