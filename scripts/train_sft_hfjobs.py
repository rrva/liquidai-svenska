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
#     "peft",
#     "pyyaml",
# ]
# ///
"""
SFT for Swedish LFM2.5-1.2B on CPT checkpoint with LoRA (Unsloth).

Based on the official Unsloth LFM2.5 SFT template, adapted for:
- Loading from our CPT checkpoint
- Reading SFT data from JSONL manifests (messages format)
- Swedish-specific configuration

Run locally:
    python scripts/train_sft_hfjobs.py --config configs/sft_1.2b.yaml

Run on HF Jobs:
    hf jobs uv run scripts/train_sft_hfjobs.py \
        --flavor a10g-small --secrets HF_TOKEN --timeout 6h \
        -- --config configs/sft_1.2b.yaml

Fallback: If Unsloth fails to load the CPT checkpoint, set --no-unsloth
to use plain transformers + trl instead.
"""

import argparse
import json
import logging
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

import yaml


def check_cuda():
    """Check CUDA availability and exit with helpful message if not available."""
    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Run on a machine with a CUDA-capable GPU or use HF Jobs:")
        logger.error("  hf jobs uv run scripts/train_sft_hfjobs.py --flavor a10g-small ...")
        sys.exit(1)
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_sft_manifest(path):
    """Load SFT JSONL manifest. Each line has {"messages": [...], "source": ...}."""
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "messages" in row and len(row["messages"]) >= 2:
                rows.append(row)
    return rows


def to_conversations_format(rows):
    """Convert messages format to Unsloth conversations format."""
    converted = []
    for row in rows:
        convs = []
        for msg in row["messages"]:
            role = msg["role"]
            # Unsloth standardize_data_formats expects "human"/"gpt" or "user"/"assistant"
            convs.append({"from": role, "value": msg["content"]})
        converted.append({"conversations": convs})
    return converted


def main():
    parser = argparse.ArgumentParser(description="SFT training for Swedish LFM2.5")
    parser.add_argument("--config", required=True, help="Path to SFT config YAML")
    parser.add_argument("--output_dir", default=None, help="Override local output dir")
    parser.add_argument("--max_steps", type=int, default=None, help="Override epochs with fixed step count (for quick tests)")
    parser.add_argument("--no_unsloth", action="store_true", help="Fallback to transformers+trl")
    parser.add_argument("--merge_model", action="store_true", help="Merge LoRA into base before upload")
    parser.add_argument("--gguf", nargs="*", default=None,
                        help="Export GGUF quantizations (Unsloth only). "
                             "Specify quantization methods e.g. --gguf q4_k_m q8_0. "
                             "Defaults to q4_k_m q8_0 if flag given with no args.")
    parser.add_argument("--no_push", action="store_true", help="Disable push to hub")
    args = parser.parse_args()

    cfg = load_config(args.config)
    check_cuda()

    # Resolve model name (prefix with HF username if needed)
    token = os.environ.get("HF_TOKEN")
    model_name = cfg["model_name"]
    output_repo = cfg.get("output_repo", "lfm25-svenska-1.2b-cpt-sft")
    output_dir = args.output_dir or f"outputs/{output_repo}"
    push_to_hub = not args.no_push

    hf_username = None
    if token:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        from huggingface_hub import HfApi, login
        login(token=token)
        hf_username = HfApi(token=token).whoami()["name"]
        # If model_name doesn't have a slash, prefix with username
        if "/" not in model_name:
            model_name = f"{hf_username}/{model_name}"
        hub_repo = f"{hf_username}/{output_repo}"
        logger.info(f"Will push to: {hub_repo}")
    else:
        hub_repo = output_repo
        if push_to_hub:
            logger.warning("HF_TOKEN not set, disabling push_to_hub")
            push_to_hub = False

    seq_length = cfg.get("seq_length", 4096)
    lora_r = cfg.get("lora_r", 16)
    lora_alpha = cfg.get("lora_alpha", 16)
    lora_dropout = cfg.get("lora_dropout", 0)
    batch_size = cfg.get("per_device_train_batch_size", 2)
    grad_accum = cfg.get("gradient_accumulation_steps", 8)
    lr = cfg.get("learning_rate", 2e-4)
    num_epochs = cfg.get("num_train_epochs", 3)
    max_steps = args.max_steps  # None means use epochs
    report_to = cfg.get("report_to", "none")

    duration_str = f"{max_steps} steps" if max_steps else f"{num_epochs} epoch(s)"

    print("=" * 70)
    print("Swedish LFM2.5-1.2B SFT (LoRA)")
    print("=" * 70)
    print(f"  Base model:     {model_name}")
    print(f"  Output:         {hub_repo}")
    print(f"  Seq length:     {seq_length}")
    print(f"  LoRA r/alpha:   {lora_r}/{lora_alpha}")
    print(f"  Batch:          {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    print(f"  LR:             {lr}")
    print(f"  Training:       {duration_str}")
    print(f"  Unsloth:        {not args.no_unsloth}")
    print()

    # Load data
    print("[1/5] Loading SFT data...")
    train_manifest = cfg.get("train_manifest", "data/manifests/sft_train.jsonl")
    eval_manifest = cfg.get("eval_manifest", "data/manifests/sft_eval.jsonl")

    train_rows = load_sft_manifest(train_manifest)
    eval_rows = load_sft_manifest(eval_manifest)
    print(f"  Train: {len(train_rows)} conversations")
    print(f"  Eval: {len(eval_rows)} conversations")

    if args.no_unsloth:
        _train_with_trl(cfg, model_name, train_rows, eval_rows, output_dir, hub_repo,
                        push_to_hub, token, seq_length, lora_r, lora_alpha, lora_dropout,
                        batch_size, grad_accum, lr, num_epochs, max_steps, report_to)
    else:
        gguf_quants = args.gguf if args.gguf else (["q4_k_m", "q8_0"] if args.gguf is not None else None)
        _train_with_unsloth(cfg, model_name, train_rows, eval_rows, output_dir, hub_repo,
                            push_to_hub, token, seq_length, lora_r, lora_alpha, lora_dropout,
                            batch_size, grad_accum, lr, num_epochs, max_steps, report_to,
                            args.merge_model, gguf_quants)


def _train_with_unsloth(cfg, model_name, train_rows, eval_rows, output_dir, hub_repo,
                         push_to_hub, _token, seq_length, lora_r, lora_alpha, lora_dropout,
                         batch_size, grad_accum, lr, num_epochs, max_steps, report_to,
                         merge_model, gguf_quants=None):
    """Train using Unsloth (preferred path)."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import standardize_data_formats, train_on_responses_only
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # Load model with Unsloth
    print("\n[2/5] Loading model with Unsloth...")
    start = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=seq_length,
        load_in_4bit=cfg.get("load_in_4bit", False),
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    # LoRA with LFM2.5-specific target modules
    # lora_dropout=0 per Unsloth recommendation (dropout unnecessary with LoRA)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print(f"  Model loaded in {time.time() - start:.1f}s")

    # Prepare datasets
    print("\n[3/5] Preparing datasets...")
    train_convs = to_conversations_format(train_rows)
    eval_convs = to_conversations_format(eval_rows)

    train_ds = Dataset.from_list(train_convs)
    eval_ds = Dataset.from_list(eval_convs) if eval_convs else None

    train_ds = standardize_data_formats(train_ds)
    if eval_ds:
        eval_ds = standardize_data_formats(eval_ds)

    def formatting_func(examples):
        texts = tokenizer.apply_chat_template(
            examples["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": [x.removeprefix(tokenizer.bos_token) for x in texts]}

    train_ds = train_ds.map(formatting_func, batched=True)
    if eval_ds:
        eval_ds = eval_ds.map(formatting_func, batched=True)

    # Configure trainer
    print("\n[4/5] Training...")
    effective_batch = batch_size * grad_accum
    steps_per_epoch = len(train_ds) // effective_batch
    logging_steps = max(1, steps_per_epoch // 10) if steps_per_epoch > 10 else 1
    save_steps = cfg.get("save_steps", max(1, steps_per_epoch // 4))
    eval_steps = cfg.get("eval_steps", save_steps)

    if max_steps:
        run_name = f"sft-svenska-{max_steps}steps"
    else:
        run_name = f"sft-svenska-{num_epochs}ep"

    training_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        num_train_epochs=num_epochs if not max_steps else 1,
        max_steps=max_steps if max_steps else -1,
        learning_rate=lr,
        logging_steps=logging_steps,
        optim="adamw_8bit",
        weight_decay=cfg.get("weight_decay", 0.01),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        seed=42,
        max_length=seq_length,
        report_to=report_to,
        run_name=run_name,
        push_to_hub=push_to_hub,
        hub_model_id=hub_repo if push_to_hub else None,
        save_steps=save_steps,
        save_total_limit=cfg.get("save_total_limit", 3),
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=eval_steps if eval_ds else None,
        load_best_model_at_end=True if eval_ds else False,
        metric_for_best_model="eval_loss",
        bf16=cfg.get("bf16", True),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_config,
    )

    # Train only on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    start = time.time()
    train_result = trainer.train()
    train_time = time.time() - start
    print(f"\n  Training done in {train_time / 60:.1f} min")

    train_loss = train_result.metrics.get("train_loss")
    if train_loss:
        print(f"  Final train loss: {train_loss:.4f}")

    if eval_ds:
        eval_results = trainer.evaluate()
        eval_loss = eval_results.get("eval_loss")
        if eval_loss:
            print(f"  Final eval loss: {eval_loss:.4f}")
            if train_loss:
                ratio = eval_loss / train_loss
                if ratio > 1.5:
                    print(f"  Warning: Eval loss is {ratio:.1f}x train loss - possible overfitting")
                else:
                    print(f"  Eval/train ratio: {ratio:.2f} - model generalizes well")

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

    # GGUF export
    if gguf_quants:
        print(f"\n[GGUF] Exporting quantizations: {gguf_quants}")
        gguf_dir = os.path.join(output_dir, "gguf")
        for quant in gguf_quants:
            print(f"  Exporting {quant}...")
            model.save_pretrained_gguf(
                gguf_dir,
                tokenizer,
                quantization_method=quant,
            )
            if push_to_hub:
                gguf_repo = f"{hub_repo}-gguf"
                model.push_to_hub_gguf(
                    gguf_repo,
                    tokenizer,
                    quantization_method=quant,
                )
                print(f"  Pushed {quant} to: {gguf_repo}")
        print(f"  GGUF files saved to: {gguf_dir}")

    # Update model card metadata
    if push_to_hub:
        from huggingface_hub import metadata_update
        try:
            metadata_update(hub_repo, {"datasets": [cfg.get("train_manifest", "custom")]}, overwrite=True)
            print(f"  Model card metadata updated")
        except Exception as e:
            logger.warning(f"Failed to update model card metadata: {e}")

    # Save run summary
    summary = {
        "model_name": model_name,
        "output": hub_repo if push_to_hub else output_dir,
        "train_conversations": len(train_rows),
        "eval_conversations": len(eval_rows),
        "epochs": num_epochs,
        "final_train_loss": train_loss,
        "framework": "unsloth",
    }
    summary_path = os.path.join(output_dir, "run_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SFT Complete ===")
    print(json.dumps(summary, indent=2))


def _train_with_trl(cfg, model_name, train_rows, eval_rows, output_dir, hub_repo,
                     push_to_hub, token, seq_length, lora_r, lora_alpha, lora_dropout,
                     batch_size, grad_accum, lr, num_epochs, max_steps, report_to):
    """Fallback: train with plain transformers + trl (no Unsloth)."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig, get_peft_model

    print("\n[2/5] Loading model (transformers fallback)...")
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

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"  Model loaded in {time.time() - start:.1f}s")

    # Prepare data as text
    print("\n[3/5] Preparing datasets...")

    def format_messages(rows):
        texts = []
        for row in rows:
            parts = []
            for msg in row["messages"]:
                role = msg["role"]
                content = msg["content"]
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            texts.append("\n".join(parts))
        return texts

    train_texts = format_messages(train_rows)
    eval_texts = format_messages(eval_rows)
    train_ds = Dataset.from_dict({"text": train_texts})
    eval_ds = Dataset.from_dict({"text": eval_texts}) if eval_texts else None

    # Train
    print("\n[4/5] Training...")
    effective_batch = batch_size * grad_accum
    steps_per_epoch = len(train_ds) // effective_batch
    save_steps = cfg.get("save_steps", max(1, steps_per_epoch // 4))

    training_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        num_train_epochs=num_epochs if not max_steps else 1,
        max_steps=max_steps if max_steps else -1,
        learning_rate=lr,
        logging_steps=max(1, steps_per_epoch // 10),
        weight_decay=cfg.get("weight_decay", 0.01),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        seed=42,
        max_length=seq_length,
        report_to=report_to,
        push_to_hub=push_to_hub,
        hub_model_id=hub_repo if push_to_hub else None,
        hub_token=token,
        save_steps=save_steps,
        save_total_limit=cfg.get("save_total_limit", 3),
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=save_steps if eval_ds else None,
        load_best_model_at_end=True if eval_ds else False,
        metric_for_best_model="eval_loss",
        bf16=cfg.get("bf16", True) and torch.cuda.is_available(),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_config,
    )

    start = time.time()
    train_result = trainer.train()
    print(f"\n  Training done in {(time.time() - start) / 60:.1f} min")

    # Save
    print("\n[5/5] Saving...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if push_to_hub:
        trainer.push_to_hub()

    summary = {
        "model_name": model_name,
        "output": hub_repo if push_to_hub else output_dir,
        "train_conversations": len(train_rows),
        "eval_conversations": len(eval_rows),
        "epochs": num_epochs,
        "final_train_loss": train_result.metrics.get("train_loss"),
        "framework": "transformers+trl",
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SFT Complete (fallback) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
