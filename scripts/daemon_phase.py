#!/usr/bin/env python3
"""Phase-tracking daemon demo (PF-LLM heartbeat analog).

The workload/path moves through phases; the daemon re-anchors on every
epoch's measurement and applies confidence-gated one-knob moves. Unlike
daemon_demo it never declares convergence: the gate runs each epoch, so a
phase change is handled by the same mechanism as initial convergence.

The world model (0.8B + LoRA adapter) is loaded once in-process; run as
root for the netns testbed with the user's HF cache:

  sudo -E env HF_HOME=$HOME/.cache/huggingface \
      .venv/bin/python scripts/daemon_phase.py
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(HERE.parent / "harness"))

import runner  # noqa: E402
import topology as topo  # noqa: E402
from config_space import RunConfig  # noqa: E402
from data import PRIMARY_METRIC, _run_metrics  # noqa: E402
from llm import extract_json  # noqa: E402
from prompts import counterfactual_prompt  # noqa: E402

OUT_DIR = HERE.parent / "results" / "policy"
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
ADAPTER = str(HERE.parent / "runs" / "ft-r2-08b-s41" / "adapter")

K_SAMPLES = 5
VOTE_MIN = 4
MARGIN = 0.15
# delay-based CC modules are blocked on the rebuilt host (allowed-CC list);
# restrict the daemon's move set to what the testbed can actually apply.
CCS_OK = ["cubic", "bbr", "reno"]
QDISCS = ["pfifo_deep", "pfifo_bdp", "fq_codel", "cake", "fq"]

# (epochs, workload, rtt_ms, loss_pct, bw_mbit)
PHASES = [
    (5, "bulk", 20, 0.5, 50),    # lossy bulk phase: default collapses
    (5, "mixed", 20, 0.0, 25),   # interactive phase: bufferbloat exposure
    (5, "bulk", 40, 1.0, 100),   # long lossy bulk phase again
]


def neighbors(cfg: RunConfig):
    out = []
    for cc in CCS_OK:
        if cc != cfg.cc:
            out.append(("cc", RunConfig(**{**cfg.__dict__, "cc": cc})))
    for qd in QDISCS:
        if qd != cfg.qdisc:
            out.append(("qdisc", RunConfig(**{**cfg.__dict__, "qdisc": qd})))
    out.append(("buf", RunConfig(**{**cfg.__dict__,
                "buf": "limited" if cfg.buf == "ample" else "ample"})))
    return out


class Model:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        m = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map="cuda")
        self.model = PeftModel.from_pretrained(m, ADAPTER)
        self.model.eval()

    def sample(self, prompt: str, n: int) -> list[str]:
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)
        ids = self.tok(text, return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=96, do_sample=True, temperature=0.8,
                num_return_sequences=n,
                pad_token_id=self.tok.eos_token_id)
        return [self.tok.decode(o[ids["input_ids"].shape[1]:],
                                skip_special_tokens=True) for o in out]


def anchor_cell(cfg: RunConfig, result: dict, metric: str) -> dict:
    d = cfg.describe()
    d["metrics"] = _run_metrics(result)
    d["primary_metric"] = metric
    d["primary_value"] = d["metrics"].get(metric)
    return d


def gated_best(model: Model, anchor: dict, nbrs, metric: str, better: str):
    want = "increase" if better == "higher" else "decrease"
    best_nb, best_gain = None, 1.0
    for knob, nb in nbrs:
        to_cell = nb.describe()
        to_cell["metrics"] = {}
        task = {"type": "counterfactual", "split": "policy", "knob": knob,
                "from_cell": anchor, "to_cell": to_cell, "metric": metric}
        prompt = counterfactual_prompt(task, "L0", [])
        samples = []
        for t in model.sample(prompt, K_SAMPLES):
            ans = extract_json(t)
            if not ans or "direction" not in ans:
                continue
            try:
                samples.append((str(ans["direction"]).lower(),
                                float(ans.get("ratio"))))
            except (TypeError, ValueError):
                continue
        votes = sum(1 for d, _ in samples if d == want)
        if len(samples) < K_SAMPLES or votes < VOTE_MIN:
            continue
        med = statistics.median(r for _, r in samples if r and r > 0)
        gain = med if better == "higher" else (1.0 / med if med else 0)
        if gain >= 1 + MARGIN and gain > best_gain:
            best_nb, best_gain = nb, gain
    return best_nb, best_gain


def main() -> None:
    print("loading world model...", flush=True)
    model = Model()
    cur_knobs = ("cubic", "pfifo_deep", "ample")  # kernel default
    trace = []
    t0 = time.time()
    topo.create()
    servers = runner.Servers()
    servers.start_rr()
    try:
        epoch = 0
        for pi, (n_ep, wl, rtt, loss, bw) in enumerate(PHASES):
            metric, better = PRIMARY_METRIC[wl]
            print(f"=== phase {pi}: {wl} rtt={rtt} loss={loss} bw={bw} "
                  f"(metric {metric}) ===", flush=True)
            for _ in range(n_ep):
                cur = RunConfig("Z", wl, cur_knobs[0], cur_knobs[1],
                                rtt, loss, bw, cur_knobs[2])
                result = None
                for attempt in range(3):
                    try:
                        result = runner.execute(cur, servers)
                        break
                    except Exception as e:
                        print(f"  epoch {epoch} attempt {attempt} failed: {e}",
                              flush=True)
                        time.sleep(4)
                if result is None:
                    epoch += 1
                    continue
                val = _run_metrics(result).get(metric)
                row = {"epoch": epoch, "phase": pi,
                       "t_s": round(time.time() - t0, 1),
                       "workload": wl,
                       "config": "/".join(cur_knobs), metric: val}
                print(f"[t={row['t_s']:7.1f}s] e{epoch} ph{pi} "
                      f"{row['config']:26s} {metric}={val}", flush=True)
                anchor = anchor_cell(cur, result, metric)
                nxt, gain = gated_best(model, anchor, neighbors(cur),
                                       metric, better)
                if nxt is not None:
                    print(f"          move -> {nxt.cc}/{nxt.qdisc}/{nxt.buf}"
                          f" (predicted {gain:.2f}x)", flush=True)
                    row["move"] = f"{nxt.cc}/{nxt.qdisc}/{nxt.buf}"
                    cur_knobs = (nxt.cc, nxt.qdisc, nxt.buf)
                trace.append(row)
                epoch += 1
    finally:
        servers.stop_all()
        topo.destroy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "daemon_phase.json").write_text(json.dumps(trace, indent=1))
    print("saved results/policy/daemon_phase.json", flush=True)


if __name__ == "__main__":
    main()
