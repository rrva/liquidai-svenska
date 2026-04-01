# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "trl==0.22.2",
#     "huggingface_hub[hf_transfer]",
#     "trackio",
#     "tensorboard",
#     "transformers==4.57.3",
#     "pyyaml",
# ]
# ///
"""
Continued pretraining (CPT) for LFM2.5-1.2B on Swedish text.

Uses Unsloth's UnslothTrainer with high-rank LoRA (r=128) including
embed_tokens + lm_head for language adaptation, matching the official
Unsloth CPT notebook. Falls back to transformers.Trainer (full-param)
if Unsloth is unavailable.

Run locally:
    python scripts/train_cpt_hfjobs.py --config configs/cpt_1.2b.yaml

Run on HF Jobs:
    hf jobs uv run scripts/train_cpt_hfjobs.py \
        --flavor a100-small --secrets HF_TOKEN --timeout 24h \
        -- --config configs/cpt_1.2b.yaml

Fallback (no Unsloth):
    python scripts/train_cpt_hfjobs.py --config configs/cpt_1.2b.yaml --no-unsloth
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

import yaml
from datasets import Dataset


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
    for i in range(0, len(all_ids) - seq_length + 1, seq_length):
        packed.append(all_ids[i : i + seq_length])

    print(f"  Total tokens: {len(all_ids):,}")
    print(f"  Packed sequences: {len(packed):,} (seq_length={seq_length})")
    return Dataset.from_dict({"input_ids": packed})


def main():
    parser = argparse.ArgumentParser(description="CPT training for Swedish LFM2.5")
    parser.add_argument("--config", required=True, help="Path to CPT config YAML")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--no_unsloth", action="store_true", help="Fallback to transformers (full-param)")
    parser.add_argument("--no_merge", action="store_true", help="Push LoRA adapter only (default: merge into base)")
    parser.add_argument("--no_push", action="store_true", help="Disable push to hub")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model_name"]
    seq_length = cfg.get("seq_length", 2048)
    output_repo = cfg.get("output_repo", "lfm25-svenska-1.2b-cpt")
    output_dir = args.output_dir or f"outputs/{output_repo}"
    push_to_hub = not args.no_push

    token = os.environ.get("HF_TOKEN")
    hf_username = None
    if token:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        from huggingface_hub import HfApi, login
        login(token=token)
        hf_username = HfApi(token=token).whoami()["name"]
        hub_repo = f"{hf_username}/{output_repo}"
        logger.info(f"Will push to: {hub_repo}")
    else:
        hub_repo = output_repo
        if push_to_hub:
            logger.warning("HF_TOKEN not set, disabling push_to_hub")
            push_to_hub = False

    batch_size = cfg.get("per_device_train_batch_size", 2)
    grad_accum = cfg.get("gradient_accumulation_steps", 8)
    lr = cfg.get("learning_rate", 5e-5)
    num_epochs = cfg.get("num_train_epochs", 1)
    report_to = cfg.get("report_to", "none")

    print("=" * 70)
    print("Swedish LFM2.5-1.2B CPT")
    print("=" * 70)
    print(f"  Model:          {model_name}")
    print(f"  Output:         {hub_repo}")
    print(f"  Seq length:     {seq_length}")
    print(f"  Batch:          {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    print(f"  LR:             {lr}")
    print(f"  Epochs:         {num_epochs}")
    print(f"  Unsloth:        {not args.no_unsloth}")
    print(f"  Merge adapter:  {not args.no_merge}")
    print()

    # Load data
    print("[1/5] Loading CPT data...")
    train_manifest = cfg.get("train_manifest", "data/manifests/cpt_train.jsonl")
    eval_manifest = cfg.get("eval_manifest", "data/manifests/cpt_eval.jsonl")

    train_texts = load_manifest(train_manifest)
    eval_texts = load_manifest(eval_manifest)
    print(f"  Train: {len(train_texts)} documents")
    print(f"  Eval: {len(eval_texts)} documents")

    if args.no_unsloth:
        _train_with_transformers(cfg, model_name, train_texts, eval_texts, output_dir,
                                 hub_repo, push_to_hub, token, seq_length, batch_size,
                                 grad_accum, lr, num_epochs, report_to)
    else:
        _train_with_unsloth(cfg, model_name, train_texts, eval_texts, output_dir,
                            hub_repo, push_to_hub, token, seq_length, batch_size,
                            grad_accum, lr, num_epochs, report_to, not args.no_merge)


def _train_with_unsloth(cfg, model_name, train_texts, eval_texts, output_dir, hub_repo,
                         push_to_hub, _token, seq_length, batch_size, grad_accum, lr,
                         num_epochs, report_to, merge_model):
    """Train using Unsloth with LoRA (preferred path, matches official notebook)."""
    # Disable CCE — not supported for CPT
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

    # Load model
    print("\n[2/5] Loading model with Unsloth...")
    start = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=seq_length,
        load_in_4bit=False,
        dtype=None,  # auto detect
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA with high rank + embed_tokens/lm_head for language adaptation
    lora_r = cfg.get("lora_r", 128)
    lora_alpha = cfg.get("lora_alpha", 32)
    lora_dropout = cfg.get("lora_dropout", 0)
    use_rslora = cfg.get("use_rslora", True)

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj",
                         "w1", "w2", "w3",
                         "embed_tokens", "lm_head"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=use_rslora,
    )
    print(f"  Model loaded in {time.time() - start:.1f}s")
    print(f"  LoRA r={lora_r}, alpha={lora_alpha}, rsLoRA={use_rslora}")

    # Prepare data — append EOS token and create text dataset
    print("\n[3/5] Preparing datasets...")
    eos = tokenizer.eos_token

    train_ds = Dataset.from_dict({"text": [t + eos for t in train_texts]})
    eval_ds = Dataset.from_dict({"text": [t + eos for t in eval_texts]}) if eval_texts else None

    print(f"  Train: {len(train_ds)} documents")
    if eval_ds:
        print(f"  Eval: {len(eval_ds)} documents")

    # Configure trainer
    print("\n[4/5] Training...")
    effective_batch = batch_size * grad_accum
    steps_per_epoch = len(train_ds) // effective_batch
    save_steps = cfg.get("save_steps", max(1, steps_per_epoch // 4))
    eval_steps = cfg.get("eval_steps", save_steps)
    embedding_lr = cfg.get("embedding_learning_rate", lr / 10)

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=seq_length,
        dataset_num_proc=4,
        args=UnslothTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_ratio=cfg.get("warmup_ratio", 0.1),
            num_train_epochs=num_epochs,
            learning_rate=lr,
            embedding_learning_rate=embedding_lr,
            logging_steps=cfg.get("logging_steps", 1),
            optim="adamw_8bit",
            weight_decay=cfg.get("weight_decay", 0.0),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
            seed=3407,
            report_to=report_to,
            run_name=f"cpt-svenska-{num_epochs}ep",
            push_to_hub=push_to_hub,
            hub_model_id=hub_repo if push_to_hub else None,
            save_steps=save_steps,
            save_total_limit=cfg.get("save_total_limit", 3),
            eval_strategy="steps" if eval_ds else "no",
            eval_steps=eval_steps if eval_ds else None,
            load_best_model_at_end=True if eval_ds else False,
            metric_for_best_model="eval_loss",
            bf16=cfg.get("bf16", True),
        ),
    )

    start = time.time()
    train_result = trainer.train()
    train_time = time.time() - start
    print(f"\n  Training done in {train_time / 60:.1f} min")

    train_loss = train_result.metrics.get("train_loss")
    if train_loss:
        print(f"  Final train loss: {train_loss:.4f}")

    eval_loss = None
    eval_ppl = None
    if eval_ds:
        eval_results = trainer.evaluate()
        eval_loss = eval_results.get("eval_loss")
        if eval_loss:
            eval_ppl = math.exp(eval_loss)
            print(f"  Final eval loss: {eval_loss:.4f}")
            print(f"  Final eval perplexity: {eval_ppl:.2f}")

    # Save
    print("\n[5/5] Saving...")
    if merge_model and push_to_hub:
        model.push_to_hub_merged(hub_repo, tokenizer=tokenizer, save_method="merged_16bit")
        print(f"  Merged model pushed to: {hub_repo}")
    else:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        if push_to_hub:
            model.push_to_hub(hub_repo, tokenizer=tokenizer)
            print(f"  Adapter pushed to: {hub_repo}")

    # Save run summary
    summary = {
        "model_name": model_name,
        "output": hub_repo if push_to_hub else output_dir,
        "train_docs": len(train_texts),
        "eval_docs": len(eval_texts),
        "epochs": num_epochs,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "use_rslora": use_rslora,
        "final_train_loss": train_loss,
        "final_eval_loss": eval_loss,
        "final_eval_perplexity": eval_ppl,
        "framework": "unsloth",
    }
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== CPT Complete ===")
    print(json.dumps(summary, indent=2))


def _train_with_transformers(cfg, model_name, train_texts, eval_texts, output_dir,
                              hub_repo, push_to_hub, token, seq_length, batch_size,
                              grad_accum, lr, num_epochs, report_to):
    """Fallback: full-parameter CPT with transformers.Trainer (no Unsloth)."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    print("\n[2/5] Loading model (transformers fallback, full-parameter)...")
    start = time.time()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded in {time.time() - start:.1f}s")
    print(f"  Total params: {param_count:,} (all trainable)")

    # Tokenize and pack
    print("\n[3/5] Tokenizing and packing...")
    print("  Train:")
    train_ds = tokenize_and_pack(train_texts, tokenizer, seq_length)
    print("  Eval:")
    eval_ds = tokenize_and_pack(eval_texts, tokenizer, seq_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train
    print("\n[4/5] Training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=cfg.get("weight_decay", 0.0),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        num_train_epochs=num_epochs,
        bf16=cfg.get("bf16", True) and torch.cuda.is_available(),
        fp16=False,
        logging_steps=cfg.get("logging_steps", 1),
        save_steps=cfg.get("save_steps", 500),
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        push_to_hub=push_to_hub,
        hub_model_id=hub_repo if push_to_hub else None,
        hub_token=token if push_to_hub else None,
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    start = time.time()
    trainer.train()
    print(f"\n  Training done in {(time.time() - start) / 60:.1f} min")

    # Save
    print("\n[5/5] Saving...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    final_eval = trainer.evaluate()
    eval_loss = final_eval.get("eval_loss")
    eval_ppl = math.exp(eval_loss) if eval_loss else None

    summary = {
        "model_name": model_name,
        "output": hub_repo if push_to_hub else output_dir,
        "train_docs": len(train_texts),
        "eval_docs": len(eval_texts),
        "epochs": num_epochs,
        "final_eval_loss": eval_loss,
        "final_eval_perplexity": eval_ppl,
        "framework": "transformers (full-param)",
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if push_to_hub:
        trainer.push_to_hub()

    print(f"\n=== CPT Complete (fallback) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
