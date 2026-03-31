# /// script
# dependencies = ["transformers>=4.45.0", "datasets", "accelerate", "trackio", "torch", "pyyaml", "huggingface_hub"]
# ///
"""
Continued pretraining (CPT) for LFM2.5-1.2B on Swedish text.
Full-parameter training using transformers.Trainer (NOT LoRA).

Run locally:
    python scripts/train_cpt_hfjobs.py --config configs/cpt_1.2b.yaml

Run on HF Jobs:
    hf jobs uv run scripts/train_cpt_hfjobs.py \
        --flavor a100-small --secrets HF_TOKEN --timeout 24h \
        -- --config configs/cpt_1.2b.yaml
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from huggingface_hub import HfApi
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_manifest(path):
    """Load JSONL manifest, return list of text strings."""
    docs = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            if text:
                docs.append(text)
    return docs


def tokenize_and_pack(texts, tokenizer, seq_length):
    """Tokenize all texts and pack into fixed-length sequences for CLM."""
    all_ids = []
    for text in texts:
        ids = tokenizer(text, truncation=False)["input_ids"]
        all_ids.extend(ids)
        all_ids.append(tokenizer.eos_token_id)

    # Pack into chunks of seq_length
    packed = []
    for i in range(0, len(all_ids) - seq_length, seq_length):
        packed.append(all_ids[i : i + seq_length])

    print(f"  Total tokens: {len(all_ids):,}")
    print(f"  Packed sequences: {len(packed):,} (seq_length={seq_length})")
    return Dataset.from_dict({"input_ids": packed})


def main():
    parser = argparse.ArgumentParser(description="CPT training for Swedish LFM2.5")
    parser.add_argument("--config", required=True, help="Path to CPT config YAML")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--push_to_hub", action="store_true", default=True)
    parser.add_argument("--no_push", action="store_true", help="Disable push to hub")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model_name"]
    seq_length = cfg.get("seq_length", 2048)
    output_repo = cfg.get("output_repo", "lfm25-svenska-1.2b-cpt")
    output_dir = args.output_dir or f"outputs/{output_repo}"
    push_to_hub = args.push_to_hub and not args.no_push

    token = os.environ.get("HF_TOKEN")
    if push_to_hub and not token:
        print("WARNING: HF_TOKEN not set, disabling push_to_hub")
        push_to_hub = False

    # Resolve HF username for output repo
    hf_username = None
    if push_to_hub and token:
        api = HfApi(token=token)
        hf_username = api.whoami()["name"]
        hub_repo = f"{hf_username}/{output_repo}"
        print(f"Will push to: {hub_repo}")

    print(f"=== CPT Training ===")
    print(f"Model: {model_name}")
    print(f"Seq length: {seq_length}")
    print(f"Output: {output_dir}")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("\n2. Loading data...")
    train_manifest = cfg.get("train_manifest", "data/manifests/cpt_train.jsonl")
    eval_manifest = cfg.get("eval_manifest", "data/manifests/cpt_eval.jsonl")

    train_texts = load_manifest(train_manifest)
    eval_texts = load_manifest(eval_manifest)
    print(f"  Train docs: {len(train_texts):,}")
    print(f"  Eval docs: {len(eval_texts):,}")

    # Tokenize and pack
    print("\n3. Tokenizing and packing...")
    print("  Train:")
    train_ds = tokenize_and_pack(train_texts, tokenizer, seq_length)
    print("  Eval:")
    eval_ds = tokenize_and_pack(eval_texts, tokenizer, seq_length)

    # Load model
    print("\n4. Loading model...")
    dtype = torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable() if cfg.get("gradient_checkpointing", True) else None

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {param_count:,}")
    print(f"  Trainable params: {trainable_count:,}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    report_to = cfg.get("report_to", "none")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.1),
        warmup_ratio=cfg.get("warmup_ratio", 0.01),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        bf16=cfg.get("bf16", True) and torch.cuda.is_available(),
        fp16=False,
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 500),
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        push_to_hub=push_to_hub,
        hub_model_id=f"{hf_username}/{output_repo}" if push_to_hub and hf_username else None,
        hub_token=token if push_to_hub else None,
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        remove_unused_columns=False,
    )

    # Trainer
    print("\n5. Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()

    # Save
    print("\n6. Saving...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save run summary
    final_eval = trainer.evaluate()
    summary = {
        "model_name": model_name,
        "output_dir": output_dir,
        "seq_length": seq_length,
        "train_docs": len(train_texts),
        "eval_docs": len(eval_texts),
        "train_sequences": len(train_ds),
        "eval_sequences": len(eval_ds),
        "final_eval_loss": final_eval.get("eval_loss"),
        "final_eval_perplexity": math.exp(final_eval.get("eval_loss", 0)) if final_eval.get("eval_loss") else None,
    }
    summary_path = Path(output_dir) / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Run summary: {summary_path}")

    if push_to_hub and hf_username:
        print(f"\n7. Pushing to hub: {hf_username}/{output_repo}")
        trainer.push_to_hub()

    print("\n=== CPT Training Complete ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
