"""Prompt construction: task framing + grounding context blocks (L0-L3)."""

from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "crawler"))

from data import config_distance, config_of, serialize_cell
from tasks import context_pool

TESTBED = """\
Testbed: Linux network namespaces in a dumbbell topology
(client -> routerA -> routerB -> server), veth links, offloads disabled.
routerA egress applies the bottleneck: HTB rate limit {bw}Mbit with the given
qdisc as child (pfifo_deep = pfifo limit 1000 pkts; pfifo_bdp = pfifo sized to
~1 BDP), or cake with built-in shaping. routerB applies symmetric netem delay
(rtt/2 per direction) and loss on the data direction. Kernel 7.0, ECN off.
Buffers: buf=ample means tcp_{{r,w}}mem max 32MB; buf=limited means max 256KB.
Workloads: bulk = single iperf3 TCP flow, 10s, goodput_mbps is receiver-side;
rr = 64B TCP request/response pingpong (latency percentiles in ms);
mixed = bulk flow AND rr flow sharing the bottleneck (rr percentiles measure
latency-under-load); short = 40 sequential 200KB flows (per-flow completion
time percentiles, incl. connection setup).
"""

ANSWER_JSON = "Answer with a single JSON object and nothing else."

N_EXEC_CONTEXT = 14
N_RAG_DOCS = 4


def exec_context_block(cells: list[dict], task: dict, anchor: dict) -> str:
    """L2: nearest measured runs from the legal context pool."""
    pool = context_pool(cells, task)
    ranked = sorted(pool, key=lambda c: config_distance(config_of(c), config_of(anchor)))
    lines = [serialize_cell(c) for c in ranked[:N_EXEC_CONTEXT]]
    return ("Measured runs from the same testbed (other configurations):\n"
            + "\n".join(f"- {l}" for l in lines))


_RETRIEVER = None


def rag_context_block(query: str) -> str:
    """L1: kernel docs / commit / netdev-thread snippets via BM25."""
    global _RETRIEVER
    if _RETRIEVER is None:
        from build_corpus import Retriever

        _RETRIEVER = Retriever(
            __import__("pathlib").Path(__file__).resolve().parent.parent
            / "crawler" / "data" / "corpus")
    hits = _RETRIEVER.search(query, k=N_RAG_DOCS)
    parts = []
    for h in hits:
        snippet = h["text"][:1200]
        parts.append(f"[{h['type']}] {h['title']}\n{snippet}")
    return ("Excerpts from Linux kernel documentation, net/ commit messages, "
            "and netdev list discussions:\n" + "\n---\n".join(parts))


def rag_query_for(task: dict, anchor: dict) -> str:
    pm = task["metric"]
    return (f"tcp {anchor['cc']} congestion control {anchor['qdisc']} qdisc "
            f"throughput latency bufferbloat loss recovery {pm} "
            f"socket buffer tcp_rmem tcp_wmem")


def grounding_blocks(level: str, cells: list[dict], task: dict, anchor: dict) -> str:
    blocks = []
    if level in ("L1", "L3"):
        blocks.append(rag_context_block(rag_query_for(task, anchor)))
    if level in ("L2", "L3"):
        blocks.append(exec_context_block(cells, task, anchor))
    return "\n\n".join(blocks)


def testbed_for(anchor: dict) -> str:
    return TESTBED.format(bw=anchor["bw_mbit"])


def ranking_prompt(task: dict, level: str, cells: list[dict]) -> str:
    anchor = task["cells"][0]
    options = sorted(task["options"].keys())
    direction = "highest (best) first" if task["better"] == "higher" else "lowest (best) first"
    ctx = grounding_blocks(level, cells, task, anchor)
    return f"""You are a model of the Linux kernel network stack. Predict execution outcomes without running anything.

{testbed_for(anchor)}
{ctx}

Fixed condition: {task['context_desc']}.
The only knob varied is `{task['vary']}` with options: {options}.

Rank ALL options by predicted {task['metric']} ({direction}).
{ANSWER_JSON}
JSON schema: {{"ranking": ["<option>", ...], "reasoning": "<one short sentence>"}}"""


def counterfactual_prompt(task: dict, level: str, cells: list[dict]) -> str:
    a, b = task["from_cell"], task["to_cell"]
    ctx = grounding_blocks(level, cells, task, b)
    return f"""You are a model of the Linux kernel network stack. Predict execution outcomes without running anything.

{testbed_for(b)}
{ctx}

A measured run on this testbed:
- {serialize_cell(a)}

Now exactly one knob changes: `{task['knob']}` goes from {a[task['knob']]} to {b[task['knob']]}.
Everything else stays the same.

Predict the effect on {task['metric']}:
1. direction: "increase" (>+10%), "decrease" (<-10%), or "same" (within ±10%)
2. ratio: your best numeric estimate of new_value/old_value

{ANSWER_JSON}
JSON schema: {{"direction": "increase|decrease|same", "ratio": <float>, "reasoning": "<one short sentence>"}}"""


def quantitative_prompt(task: dict, level: str, cells: list[dict]) -> str:
    t = task["target"]
    ctx = grounding_blocks(level, cells, task, t)
    return f"""You are a model of the Linux kernel network stack. Predict execution outcomes without running anything.

{testbed_for(t)}
{ctx}

Target configuration (never executed for you):
- {serialize_cell(t, with_metrics=False)}

Predict {task['metric']} for this configuration.
{ANSWER_JSON}
JSON schema: {{"value": <float>, "reasoning": "<one short sentence>"}}"""


BUILDERS = {
    "ranking": ranking_prompt,
    "counterfactual": counterfactual_prompt,
    "quantitative": quantitative_prompt,
}
