#!/usr/bin/env python3
"""Batch-generate answers for exported eval prompts with a (LoRA) model.

Runs on the GPU box:
  python batch_generate.py --model Qwen/Qwen3.5-2B --adapter runs/ft-2b/adapter \
      --prompts ft/eval_prompts.jsonl --out gen-ft-2b.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompts", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--samples", type=int, default=1,
                    help="generations per prompt (>1 implies sampling)")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="only used when --samples > 1")
    args = ap.parse_args()

    # the adapter dir carries the (possibly extended) tokenizer; use it so
    # marker special tokens resolve, and resize embeddings before PEFT load
    tok_src = args.adapter if args.adapter else args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    # training may have resized (even shrunk, when the base vocab was
    # padded); match the adapter's embedding size exactly
    if args.adapter and len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in args.prompts.read_text().splitlines()]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["task_id"] for l in args.out.read_text().splitlines()}
        rows = [r for r in rows if r["task_id"] not in done]
    print(f"{len(rows)} prompts to answer ({len(done)} already done)")

    t0 = time.time()
    with open(args.out, "a") as f:
        for i, r in enumerate(rows):
            enc = tokenizer.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                add_generation_prompt=True, return_tensors="pt",
                return_dict=True).to("cuda")
            gen_kwargs = dict(max_new_tokens=args.max_new,
                              pad_token_id=tokenizer.eos_token_id)
            if args.samples > 1:
                gen_kwargs.update(do_sample=True, temperature=args.temperature,
                                  num_return_sequences=args.samples)
            else:
                gen_kwargs["do_sample"] = False
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
            texts = [tokenizer.decode(o[enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True) for o in out]
            row = {"task_id": r["task_id"], "text": texts[0]}
            if args.samples > 1:
                row["texts"] = texts
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"{i + 1}/{len(rows)} ({rate:.1f}/s)", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
