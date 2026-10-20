#!/usr/bin/env python3
"""Pantheon with observable inputs only, and replay on real paths.

  train-base            anchor-only training set from the model's corpus cells: prompts carry no
                        rtt / loss / bandwidth, only the measured run; knobs cc, qdisc, buf;
                        metrics: the primary metric, and srtt for workloads with a bulk flow
  prompts               Pantheon tasks and the corpus pool in the same serialization
  replay --round R      base set + R x 100 Pantheon tasks from the deployment nodes (4 of 17)
  score --gen F --name N   direction accuracy overall / held-out nodes / slices
  baselines             priors and RF/GBT trained on the same replayed tasks; model slices vs the forest
"""
from __future__ import annotations
import argparse, json, math, pathlib, random, statistics as st, sys
HERE = pathlib.Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(REPO / "eval")); sys.path.insert(0, str(HERE))
import os
TAG = os.environ.get("PANTHEON_TAG", "")
OUT = REPO / "results" / ("pantheon_diet" + TAG); DS = REPO / "dataset" / ("ft_diet" + TAG)
DEP_CELL = int(os.environ.get("PANTHEON_DEP_CELL", "1")); DEP_WIRED = int(os.environ.get("PANTHEON_DEP_WIRED", "3"))
OVERSAMPLE = int(os.environ.get("PANTHEON_OVERSAMPLE", "4")); FLOWS_IN_PROMPT = os.environ.get("PANTHEON_FLOWS", "") == "1"
PAIRS = os.environ.get("PANTHEON_PAIRS", "") == "1"  # replay every directed pair among cubic/bbr/vegas on a replayed path, not only cubic->x
EXCLUDE = {x for x in os.environ.get("PANTHEON_EXCLUDE", "").split(",") if x}  # nodes dropped from the evaluation set (e.g. fewer than 60 test cases)
FOLD = os.environ.get("PANTHEON_FOLD"); NFOLDS = int(os.environ.get("PANTHEON_NFOLDS", "6"))  # cross-validation over nodes: fold k holds out nodes[k::NFOLDS]
KEEP = ["goodput_mbps", "retransmits", "rr_p50_ms", "rr_p99_ms", "fct_p50_ms", "fct_p95_ms",
        "cwnd_mid", "srtt_mid_ms", "qdisc_drops", "qdisc_backlog_kb_mid"]
PRE = """You are a model of the Linux kernel network stack. Predict execution outcomes without running anything.

Setting: a sender and a receiver across a path with one bottleneck. The bottleneck has a rate limit
and a queue (pfifo_deep = pfifo limit 1000 pkts; pfifo_bdp = pfifo sized to ~1 BDP; fq, fq_codel,
cake; unknown = not visible to the sender). The round trip time, the loss rate and the bottleneck
rate of the path are NOT given: infer them from the measured run. Kernel 6.x, ECN off.
Buffers: buf=ample means tcp_{r,w}mem max 32MB; buf=limited means max 256KB.
Workloads: bulk = one or more long TCP flows, goodput_mbps is receiver-side; rr = 64B TCP
request/response pingpong (latency percentiles in ms); mixed = bulk flow AND rr flow sharing the
bottleneck; short = 40 sequential 200KB flows (completion time percentiles).
"""


def sig(x, d=3): return 0.0 if x == 0 else round(x, max(0, d - 1 - math.floor(math.log10(abs(x)))))
def direction(r): return "same" if 0.9 <= r <= 1.1 else "increase" if r > 1.1 else "decrease"


def ser(c, hide_q=False):
    ms = " ".join(f"{k}={c['metrics'][k]}" for k in KEEP if k in c["metrics"])
    return f"workload={c['workload']} cc={c['cc']} qdisc={'unknown' if hide_q else c['qdisc']} buf={c['buf']} -> {ms}"


def prompt(a, knob, new, metric, hide_q=False):
    return f"""{PRE}
A measured run on this path:
- {ser(a, hide_q)}

Now exactly one knob changes: `{knob}` goes from {a[knob]} to {new}.
Everything else stays the same.

Predict the effect on {metric}:
1. direction: "increase" (>+10%), "decrease" (<-10%), or "same" (within ±10%)
2. ratio: your best numeric estimate of new_value/old_value

Answer with a single JSON object and nothing else.
JSON schema: {{"direction": "increase|decrease|same", "ratio": <float>}}"""


def example(p, ratio):
    return {"kind": "counterfactual", "messages": [{"role": "user", "content": p},
            {"role": "assistant", "content": json.dumps({"direction": direction(ratio), "ratio": sig(ratio)})}]}


def train_base():
    import regressor_baseline as rb
    rb.MANIFEST = json.load(open(REPO / "dataset" / "ft" / "manifest.json"))
    rb.FAMILIES = {c["family"] for c in rb.load_cells(rb.RUNS_DIR)}
    cells = rb.train_cells(); rng = random.Random(42); ex = []
    for a in cells:
        for b in cells:
            if a is b or a["workload"] != b["workload"]: continue
            if any(a[k] != b[k] for k in ("rtt_ms", "loss_pct", "bw_mbit")): continue
            d = [k for k in ("cc", "qdisc", "buf") if a[k] != b[k]]
            if len(d) != 1: continue
            metrics = [a["primary_metric"]] + (["srtt_mid_ms"] if a["workload"] in ("bulk", "mixed") else [])
            for m in metrics:
                va, vb = a["metrics"].get(m), b["metrics"].get(m)
                if not va or not vb or va <= 0: continue
                hide = d[0] != "qdisc" and rng.random() < 0.3
                ex.append(example(prompt(a, d[0], b[d[0]], m, hide), vb / va))
    rng.shuffle(ex); n = max(1, int(0.05 * len(ex)))
    out = DS / "base"; out.mkdir(parents=True, exist_ok=True)
    (out / "val.jsonl").write_text("\n".join(json.dumps(e) for e in ex[:n]) + "\n")
    (out / "train.jsonl").write_text("\n".join(json.dumps(e) for e in ex[n:]) + "\n")
    print("anchor-only examples:", len(ex), "train", len(ex) - n)


def pantheon_tasks():
    ts = [json.loads(l) for l in open(REPO / "dataset" / "pantheon" / "tasks.jsonl")]
    nodes = sorted({t["node"] for t in ts}); rng = random.Random(7); rng.shuffle(nodes)  # fold assignment stays as before the exclusion
    if EXCLUDE: ts = [t for t in ts if t["node"] not in EXCLUDE]
    cell = {t["node"]: t["kind"] for t in ts}
    if FOLD is not None:
        held_nodes = nodes[int(FOLD)::NFOLDS]; dep = [n for n in nodes if n not in held_nodes]
    else:
        dep = [n for n in nodes if cell[n] == "cellular"][:DEP_CELL] + [n for n in nodes if cell[n] != "cellular"][:DEP_WIRED]
    for t in ts:
        a = t["anchor"]
        t["half"] = "deploy" if t["node"] in dep else "heldout"
        retx = int(round(a["loss"] * a["tput"] * 1e6 / 8 / 1500 * 10))
        wl = "bulk" if not FLOWS_IN_PROMPT or t["flows"] == 1 else f"bulk({t['flows']} flows)"
        c = {"workload": wl, "cc": "cubic", "qdisc": "unknown", "buf": "ample",
             "metrics": {"goodput_mbps": round(a["tput"], 3), "retransmits": retx, "srtt_mid_ms": round(2 * a["delay"], 3)}}
        t["metric_name"] = "goodput_mbps" if t["metric"] == "tput" else "srtt_mid_ms"
        t["diet_prompt"] = prompt(c, "cc", t["target"], t["metric_name"], hide_q=True)
    return ts, dep


def prompts():
    OUT.mkdir(parents=True, exist_ok=True)
    ts, dep = pantheon_tasks(); print("deployment nodes:", dep, "tasks", sum(t["half"] == "deploy" for t in ts), "held-out", sum(t["half"] == "heldout" for t in ts))
    with open(OUT / "pantheon_prompts_diet.jsonl", "w") as f:
        for t in ts: f.write(json.dumps({"task_id": t["task_id"], "prompt": t["diet_prompt"]}) + "\n")
    with open(OUT / "pool_prompts_diet.jsonl", "w") as f:
        for t in (map(json.loads, open(REPO / "results" / "flywheel_suite" / "pool.jsonl")) if (REPO / "results" / "flywheel_suite" / "pool.jsonl").exists() else []):  # optional corpus pool, not shipped
            f.write(json.dumps({"task_id": t["task_id"], "prompt": prompt(t["from_cell"], t["knob"], t["to_cell"][t["knob"]], t["metric"])}) + "\n")


def replay(rnd: int, k: int = 100, oversample: int = OVERSAMPLE):
    ts, _ = pantheon_tasks(); dep = [t for t in ts if t["half"] == "deploy"]
    random.Random(11).shuffle(dep); picked = dep[: rnd * k]
    base = (REPO / "dataset" / "ft_diet" / "base" / "train.jsonl").read_text().splitlines()
    new = [json.dumps(example(t["diet_prompt"], t["truth_ratio"])) for t in picked]
    if PAIRS:
        new = pair_examples(picked)
    rows = base + new * oversample; random.Random(rnd).shuffle(rows)
    out = DS / f"r{rnd}"; out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("\n".join(rows) + "\n"); (out / "val.jsonl").write_text((REPO / "dataset" / "ft_diet" / "base" / "val.jsonl").read_text())
    print(f"round {rnd}: {len(base)} + {len(new)}x{oversample}")


def score(gen, name, pool=False):
    from llm import extract_json
    gens = {g["task_id"]: g["text"] for g in map(json.loads, open(gen))}
    if pool:
        ts = [json.loads(l) for l in open(REPO / "results" / "flywheel_suite" / "pool.jsonl")]
        ok = [int(str((extract_json(gens[t["task_id"]]) or {}).get("direction", "")).lower() == t["truth_direction"]) for t in ts]
        res = {"name": name, "suite_pool": round(st.mean(ok), 4), "n": len(ok)}
    else:
        ts, dep = pantheon_tasks(); rows = []
        for t in ts:
            a = extract_json(gens.get(t["task_id"], "")) or {}
            rows.append((t, int(str(a.get("direction", "")).lower() == t["truth_direction"])))
        def acc(f): 
            v = [o for t, o in rows if f(t)]; return (round(st.mean(v), 4), len(v)) if v else None
        res = {"name": name, "all": acc(lambda t: True), "heldout_nodes": acc(lambda t: t["half"] == "heldout"),
               "deploy_nodes": acc(lambda t: t["half"] == "deploy"),
               "heldout_tput": acc(lambda t: t["half"] == "heldout" and t["metric"] == "tput"),
               "heldout_delay": acc(lambda t: t["half"] == "heldout" and t["metric"] == "delay"),
               "heldout_cellular": acc(lambda t: t["half"] == "heldout" and t["kind"] == "cellular"),
               "heldout_wired": acc(lambda t: t["half"] == "heldout" and t["kind"] != "cellular"),
               "heldout_1flow": acc(lambda t: t["half"] == "heldout" and t["flows"] == 1)}
        held = [t for t in ts if t["half"] == "heldout"]
        res["heldout_nochange"] = round(st.mean(t["truth_direction"] == "same" for t in held), 4)
        import collections
        maj = collections.Counter((t["target"], t["metric"], t["truth_direction"]) for t in ts if t["half"] == "deploy")
        best = {}
        for (tg, m, d), n in maj.items():
            if n > best.get((tg, m), (None, 0))[1]: best[(tg, m)] = (d, n)
        res["heldout_prior_from_deploy_nodes"] = round(st.mean(best.get((t["target"], t["metric"]), ("same", 0))[0] == t["truth_direction"] for t in held), 4)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f"score_{name}.json").write_text(json.dumps(res, indent=1)); print(res, flush=True)


def baselines(k: int = 300):
    """Priors and tabular learners that see the same replayed tasks as the model, plus per-model slices
    (deployment-node tasks never replayed; paired test against the forest). -> score_baselines.json"""
    import collections
    from llm import extract_json
    from math import comb
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    ts, _ = pantheon_tasks(); dep = [t for t in ts if t["half"] == "deploy"]; held = [t for t in ts if t["half"] == "heldout"]
    random.Random(11).shuffle(dep); rep, unrep = dep[:k], dep[k:]
    def prior(src, tgt):
        maj = collections.Counter((t["target"], t["metric"], t["truth_direction"]) for t in src); best = {}
        for (tg, m, d), n in maj.items():
            if n > best.get((tg, m), (None, 0))[1]: best[(tg, m)] = (d, n)
        return round(st.mean(best.get((t["target"], t["metric"]), ("same", 0))[0] == t["truth_direction"] for t in tgt), 4)
    def wilson(p, n, z=1.96):
        d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return round(c - h, 3), round(c + h, 3)
    res = {"k_replayed": k, "n_heldout": len(held), "n_deploy_unreplayed": len(unrep),
           "nochange": {"heldout": round(st.mean(t["truth_direction"] == "same" for t in held), 4),
                        "deploy_unreplayed": round(st.mean(t["truth_direction"] == "same" for t in unrep), 4) if unrep else None},
           "prior_from_replayed": {"heldout": prior(rep, held), "deploy_unreplayed": prior(rep, unrep) if unrep else None},
           "prior_from_all_deploy": {"heldout": prior(dep, held)},
           "prior_leave_one_node_out": {"heldout": round(st.mean(prior([u for u in ts if u["node"] != t["node"]], [t]) for t in held), 4)}}
    # tabular learners on the model-visible anchor: throughput, delay, loss, target cc, metric
    def X(sel): return np.array([[t["anchor"]["tput"], t["anchor"]["delay"], t["anchor"]["loss"] * 1e3, t["target"] == "bbr", t["metric"] == "tput"] for t in sel], float)
    def y(sel): return [t["truth_direction"] for t in sel]
    rf_pred = {}; pair_pred = {}
    for name, mk in [("rf", lambda s: RandomForestClassifier(300, random_state=s, min_samples_leaf=3)), ("gbt", lambda s: GradientBoostingClassifier(random_state=s))]:
        for src, lab in [(rep, "replayed"), (dep, "all_deploy")]:
            accs = collections.defaultdict(list)
            for s in range(5):
                clf = mk(s).fit(X(src), y(src))
                ph = clf.predict(X(held)); accs["heldout"].append(st.mean(p == t for p, t in zip(ph, y(held))))
                if lab == "replayed" and name == "rf" and s == 0: rf_pred = dict(zip((t["task_id"] for t in held), ph))
                if lab == "replayed" and unrep:
                    accs["deploy_unreplayed"].append(st.mean(p == t for p, t in zip(clf.predict(X(unrep)), y(unrep))))
            res[f"{name}_from_{lab}"] = {kk: (round(st.mean(v), 4), round(min(v), 4), round(max(v), 4)) for kk, v in accs.items()}
    if PAIRS:
        # fair tabular baselines for pair augmentation: all directed cubic/bbr/vegas pairs of the replayed paths,
        # from- and to-scheme as features, tested on the held-out cubic->x tasks
        from pantheon_tasks import scheme_mean
        SC = ["cubic", "bbr", "vegas"]
        def prow(m, a, b, metric, flows): return [m["tput"], m["delay"], m["loss"] * 1e3] + [a == c for c in SC] + [b == c for c in SC] + [metric == "tput", flows]
        groups = {}
        for t in rep: groups.setdefault((t["file"], t["flows"]), t)
        Xp, yp = [], []
        for (fn, flows), t in groups.items():
            exp = json.load(open(REPO / "dataset" / "pantheon" / "raw" / fn)); means = {c: scheme_mean(exp, c) for c in SC}; means = {c: m for c, m in means.items() if m}
            for a in means:
                for b in means:
                    if a == b: continue
                    for metric in ("tput", "delay"):
                        Xp.append(prow(means[a], a, b, metric, flows)); yp.append(direction(means[b][metric] / means[a][metric]))
        Xp = np.array(Xp, float); Xph = np.array([prow(t["anchor"], "cubic", t["target"], t["metric"], t["flows"]) for t in held], float)
        res["pair_rows"] = len(Xp)
        for name, mk in [("rf", lambda s: RandomForestClassifier(300, random_state=s, min_samples_leaf=3)), ("gbt", lambda s: GradientBoostingClassifier(random_state=s))]:
            accs = []
            for s in range(5):
                clf = mk(s).fit(Xp, yp); ph = clf.predict(Xph); accs.append(st.mean(p == t for p, t in zip(ph, y(held))))
                if s == 0: pair_pred[name] = dict(zip((t["task_id"] for t in held), ph))
            res[f"{name}_from_all_pairs"] = {"heldout": (round(st.mean(accs), 4), round(min(accs), 4), round(max(accs), 4))}
    # model slices
    models = {}
    for f in sorted(OUT.glob("gen_diet*.jsonl")):
        gens = {g["task_id"]: str((extract_json(g["text"]) or {}).get("direction", "")).lower() for g in map(json.loads, open(f))}
        ok = lambda t: gens.get(t["task_id"]) == t["truth_direction"]
        ph = st.mean(ok(t) for t in held)
        mw = sum(ok(t) and rf_pred[t["task_id"]] != t["truth_direction"] for t in held)
        rw = sum((not ok(t)) and rf_pred[t["task_id"]] == t["truth_direction"] for t in held)
        n = mw + rw; p = min(1.0, 2 * sum(comb(n, i) for i in range(min(mw, rw) + 1)) / 2 ** n) if n else 1.0
        vs_pairs = {}
        for name, pp in pair_pred.items():
            a = sum(ok(t) and pp[t["task_id"]] != t["truth_direction"] for t in held); b = sum((not ok(t)) and pp[t["task_id"]] == t["truth_direction"] for t in held)
            nn = a + b; vs_pairs[f"vs_{name}_pairs"] = {"model_right_other_wrong": a, "other_right_model_wrong": b, "p": round(min(1.0, 2 * sum(comb(nn, i) for i in range(min(a, b) + 1)) / 2 ** nn) if nn else 1.0, 4)}
        models[f.stem[4:]] = {"heldout": (round(ph, 4), wilson(ph, len(held))), "deploy_unreplayed": round(st.mean(ok(t) for t in unrep), 4) if unrep else None, **vs_pairs,
                              "deploy_replayed": round(st.mean(ok(t) for t in rep), 4),
                              "vs_rf_heldout": {"model_right_rf_wrong": mw, "rf_right_model_wrong": rw, "p": round(p, 4)},
                              "answers_heldout": dict(collections.Counter(gens.get(t["task_id"]) for t in held))}
    res["models"] = models
    (OUT / ("score_baselines.json" if k == 300 else f"score_baselines_k{k}.json")).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def anchor_cfg(m, flows, hide_q=True):
    retx = int(round(m["loss"] * m["tput"] * 1e6 / 8 / 1500 * 10))
    wl = "bulk" if not FLOWS_IN_PROMPT or flows == 1 else f"bulk({flows} flows)"
    return {"workload": wl, "cc": m["cc"], "qdisc": "unknown", "buf": "ample",
            "metrics": {"goodput_mbps": round(m["tput"], 3), "retransmits": retx, "srtt_mid_ms": round(2 * m["delay"], 3)}}


def pair_examples(picked):
    """All directed one-knob pairs among the kernel schemes measured on the replayed paths (cubic, bbr, vegas),
    for both metrics: the same measurements as the cubic->x tasks, serialized from every anchor."""
    from pantheon_tasks import scheme_mean
    groups = {}
    for t in picked: groups.setdefault((t["file"], t["flows"]), t)
    ex = []
    for (fn, flows), t in groups.items():
        exp = json.load(open(REPO / "dataset" / "pantheon" / "raw" / fn))
        means = {sc: scheme_mean(exp, sc) for sc in ("cubic", "bbr", "vegas")}
        means = {sc: dict(m, cc=sc) for sc, m in means.items() if m}
        for a in means:
            for b in means:
                if a == b: continue
                for metric, key in (("goodput_mbps", "tput"), ("srtt_mid_ms", "delay")):
                    ex.append(json.dumps(example(prompt(anchor_cfg(means[a], flows), "cc", b, metric, hide_q=True), means[b][key] / means[a][key])))
    return ex


def cvagg(prefix="dietcv", nfolds=NFOLDS, tag="cv"):
    """Pool the held-out answers of all folds: overall accuracy for the fold model, the base model and the
    same-data RF, per node and per (target, metric). -> results/pantheon_diet_cv/score_cvagg.json"""
    import collections
    rows = []
    for k in range(nfolds):
        d = REPO / "results" / f"pantheon_diet_{tag}{k}"
        bf = next(iter(sorted(d.glob("score_baselines*.json"))), None); b = json.load(open(bf)) if bf else {}
        rows.append((k, json.load(open(d / f"score_{prefix}_{k}.json")), json.load(open(d / "score_diet_base.json")), b))
    n = sum(r[1]["heldout_nodes"][1] for r in rows)
    acc = lambda key: round(sum(r[1][key][0] * r[1][key][1] for r in rows if r[1][key]) / sum(r[1][key][1] for r in rows if r[1][key]), 4)
    res = {"n_heldout_total": n, "model": {k: acc(k) for k in ("heldout_nodes", "heldout_tput", "heldout_delay", "heldout_cellular", "heldout_wired")},
           "base": round(sum(r[2]["heldout_nodes"][0] * r[2]["heldout_nodes"][1] for r in rows) / n, 4),
           "rf_same_data": round(sum(r[3]["rf_from_replayed"]["heldout"][0] * r[1]["heldout_nodes"][1] for r in rows if r[3]) / n, 4) if all(r[3] for r in rows) else None,
           "gbt_same_data": round(sum(r[3]["gbt_from_replayed"]["heldout"][0] * r[1]["heldout_nodes"][1] for r in rows if r[3]) / n, 4) if all(r[3] for r in rows) else None,
           "rf_all_pairs": round(sum(r[3]["rf_from_all_pairs"]["heldout"][0] * r[1]["heldout_nodes"][1] for r in rows) / n, 4) if all(r[3] and "rf_from_all_pairs" in r[3] for r in rows) else None,
           "gbt_all_pairs": round(sum(r[3]["gbt_from_all_pairs"]["heldout"][0] * r[1]["heldout_nodes"][1] for r in rows) / n, 4) if all(r[3] and "gbt_from_all_pairs" in r[3] for r in rows) else None,
           "nochange": round(sum(r[1]["heldout_nochange"] * r[1]["heldout_nodes"][1] for r in rows) / n, 4),
           "prior_from_deploy": round(sum(r[1]["heldout_prior_from_deploy_nodes"] * r[1]["heldout_nodes"][1] for r in rows) / n, 4),
           "folds": {r[0]: {"model": r[1]["heldout_nodes"], "base": r[2]["heldout_nodes"][0], "rf": r[3].get("rf_from_replayed", {}).get("heldout") if r[3] else None} for r in rows}}
    out = REPO / "results" / f"pantheon_diet_{tag}"; out.mkdir(exist_ok=True)
    (out / "score_cvagg.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def decide(gen=None, samples=None, name="", k=300, vote_min=4, margin=0.15, rf_conf=0.8):
    """Decision-level evaluation on the held-out paths. A decision unit is (report file, flows, objective metric):
    the operator wants more throughput or less delay and may keep CUBIC or move to BBR or Vegas. Gain of a
    scheme = truth ratio (tput) or 1/ratio (delay); optimum = best of the three. Policies: hold, always-bbr,
    always-vegas, model argmax (greedy answer), model gated (>= vote_min of K samples say improve and the
    geometric-mean predicted gain over those votes clears the margin), RF argmax and RF gated (proba >= rf_conf)
    trained on the same k replayed tasks. -> score_decide_<name>.json"""
    import collections
    from llm import extract_json
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    ts, _ = pantheon_tasks(); dep = [t for t in ts if t["half"] == "deploy"]; held = [t for t in ts if t["half"] == "heldout"]
    random.Random(11).shuffle(dep); rep = dep[:k]
    better = lambda t: "increase" if t["metric"] == "tput" else "decrease"
    def gain_of(t, ratio):
        return ratio if t["metric"] == "tput" else (1 / ratio if ratio > 0 else 1.0)
    # model answers
    greedy = {}
    if gen:
        for g in map(json.loads, open(gen)):
            a = extract_json(g["text"]) or {}
            try: greedy[g["task_id"]] = (str(a.get("direction", "")).lower(), float(a.get("ratio")))
            except (TypeError, ValueError): greedy[g["task_id"]] = (str(a.get("direction", "")).lower(), None)
    samp = {}
    if samples:
        for g in map(json.loads, open(samples)):
            outs = []
            for txt in g["samples"]:
                a = extract_json(txt) or {}
                try: outs.append((str(a.get("direction", "")).lower(), float(a.get("ratio"))))
                except (TypeError, ValueError): outs.append((str(a.get("direction", "")).lower(), None))
            samp[g["task_id"]] = outs
    # RF on the same replayed tasks, with probabilities
    def X(sel): return np.array([[t["anchor"]["tput"], t["anchor"]["delay"], t["anchor"]["loss"] * 1e3, t["target"] == "bbr", t["metric"] == "tput"] for t in sel], float)
    rf = RandomForestClassifier(300, random_state=0, min_samples_leaf=3).fit(X(rep), [t["truth_direction"] for t in rep])
    proba = rf.predict_proba(X(held)); cls = list(rf.classes_)
    rf_p = {t["task_id"]: dict(zip(cls, pr)) for t, pr in zip(held, proba)}
    # decision units
    units = collections.defaultdict(list)
    for t in held: units[(t["file"], t["flows"], t["metric"])].append(t)
    def pred_gain_greedy(t):
        d, r = greedy.get(t["task_id"], ("", None))
        if d != better(t) or not r or r <= 0: return None
        return gain_of(t, r)
    def pred_gain_gated(t):
        outs = samp.get(t["task_id"], [])
        votes = [r for d, r in outs if d == better(t) and r and r > 0]
        if len(outs) < vote_min or len(votes) < vote_min: return None
        g = math.exp(st.mean(math.log(gain_of(t, r)) for r in votes))
        return g if g >= 1 + margin else None
    def pred_gain_greedy_margin(t):
        g = pred_gain_greedy(t); return g if g and g >= 1 + margin else None
    def pred_gain_rf(t, conf):
        p = rf_p[t["task_id"]].get(better(t), 0.0)
        if p < conf or max(rf_p[t["task_id"]], key=rf_p[t["task_id"]].get) != better(t): return None
        return 1.0 + p
    policies = {"hold": lambda t: None,
                "always_bbr": lambda t: 2.0 if t["target"] == "bbr" else None,
                "always_vegas": lambda t: 2.0 if t["target"] == "vegas" else None,
                "rf_argmax": lambda t: pred_gain_rf(t, 0.0), f"rf_conf{rf_conf}": lambda t: pred_gain_rf(t, rf_conf)}
    if greedy: policies["model_argmax"] = pred_gain_greedy; policies[f"model_greedy_margin{margin}"] = pred_gain_greedy_margin
    if samp: policies[f"model_gated_{vote_min}of5_m{margin}"] = pred_gain_gated
    res = {"name": name, "k_replayed": k, "units": len(units), "policies": {}}
    for pname, f in policies.items():
        shares, moves, helped, hurt, below = [], 0, 0, 0, 0
        for key, cands in units.items():
            truth = {t["target"]: gain_of(t, t["truth_ratio"]) for t in cands}
            opt = max(1.0, max(truth.values()))
            preds = {t["target"]: f(t) for t in cands}; preds = {a: g for a, g in preds.items() if g}
            if preds:
                choice = max(preds, key=preds.get); out = truth[choice]; moves += 1
                helped += out > 1.1; hurt += out < 0.9
            else:
                out = 1.0
            shares.append(out / opt); below += out < 0.9
        res["policies"][pname] = {"mean_share": round(st.mean(shares), 4), "median_share": round(st.median(shares), 4),
                                  "below_cubic": below, "moves": moves, "moves_helped": helped, "moves_hurt": hurt,
                                  "share_lt_0.5": sum(s < 0.5 for s in shares)}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f"score_decide_{name}.json").write_text(json.dumps(res, indent=1))
    for pname, v in res["policies"].items(): print(f"{name:14s} {pname:28s} share {v['mean_share']:.3f} below {v['below_cubic']:3d} moves {v['moves']:4d} helped {v['moves_helped']:4d} hurt {v['moves_hurt']:3d}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--round", type=int)
    ap.add_argument("--gen"); ap.add_argument("--name"); ap.add_argument("--pool", action="store_true"); ap.add_argument("--k", type=int, default=300); ap.add_argument("--samples")
    a = ap.parse_args()
    {"train-base": train_base, "prompts": prompts, "replay": lambda: replay(a.round, a.k if a.k != 300 else 100), "score": lambda: score(a.gen, a.name, a.pool),
     "baselines": lambda: baselines(a.k), "decide": lambda: decide(a.gen, a.samples, a.name, a.k), "cvagg": lambda: cvagg(a.name or "dietcv", NFOLDS, os.environ.get("PANTHEON_CVTAG", "cv"))}[a.stage]()
