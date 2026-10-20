#!/usr/bin/env python3
"""LoRA fine-tune a small Qwen on the netmodel FT dataset.

Loss is masked to the assistant completion (the JSON answer) only.

  python train_lora.py --model Qwen/Qwen3.5-2B --data dataset/ft --out runs/ft-2b
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments, set_seed,
)


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()]


def encode(example: dict, tokenizer, max_len: int) -> dict:
    msgs = example["messages"]
    # tokenize prompt and completion separately: BPE merges across the
    # boundary otherwise break the token-level prefix property
    p_str = tokenizer.apply_chat_template(
        msgs[:-1], add_generation_prompt=True, tokenize=False)
    f_str = tokenizer.apply_chat_template(msgs, tokenize=False)
    assert f_str.startswith(p_str), "template prefix mismatch"
    prompt_ids = tokenizer(p_str, add_special_tokens=False).input_ids
    comp_ids = tokenizer(f_str[len(p_str):], add_special_tokens=False).input_ids
    full_ids = prompt_ids + comp_ids
    labels = [-100] * len(prompt_ids) + comp_ids
    if len(full_ids) > max_len:
        raise ValueError(f"example too long: {len(full_ids)} > {max_len}")
    return {"input_ids": full_ids, "labels": labels,
            "attention_mask": [1] * len(full_ids)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--data", default="dataset/ft")
    ap.add_argument("--out", default="runs/ft-2b")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="also save a merged full model for vLLM serving")
    ap.add_argument("--special-tokens", default=None,
                    help="comma-separated marker tokens to add to the "
                         "vocabulary (PF-LLM style); embeddings become "
                         "trainable via modules_to_save")
    args = ap.parse_args()
    set_seed(args.seed)

    data_dir = pathlib.Path(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False

    modules_to_save = None
    if args.special_tokens:
        toks = [s.strip() for s in args.special_tokens.split(",") if s.strip()]
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": toks})
        model.resize_token_embeddings(len(tokenizer))
        tied = getattr(model.config, "tie_word_embeddings", False)
        modules_to_save = ["embed_tokens"] + ([] if tied else ["lm_head"])
        print(f"added {added} special tokens; trainable embeddings via "
              f"modules_to_save={modules_to_save} (tied={tied})")

    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if args.grad_ckpt:
        model.enable_input_require_grads()

    def prep(split: str) -> Dataset:
        rows = load_jsonl(data_dir / f"{split}.jsonl")
        enc = [encode(r, tokenizer, args.max_len) for r in rows]
        return Dataset.from_list(enc)

    train_ds, val_ds = prep("train"), prep("val")
    lens = [len(x["input_ids"]) for x in train_ds]
    print(f"train={len(train_ds)} val={len(val_ds)} "
          f"tok len p50={sorted(lens)[len(lens)//2]} max={max(lens)}")

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True,
                                      label_pad_token_id=-100)
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        bf16=True,
        gradient_checkpointing=args.grad_ckpt,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator)
    trainer.train()
    metrics = trainer.evaluate()
    print("final val:", metrics)

    adapter_dir = pathlib.Path(args.out) / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print("adapter ->", adapter_dir)

    if args.merge:
        merged = model.merge_and_unload()
        merged_dir = pathlib.Path(args.out) / "merged"
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print("merged ->", merged_dir)


if __name__ == "__main__":
    main()
