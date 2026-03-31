"""
Evaluate perplexity: compare base model vs CPT and/or SFT checkpoints on held-out Swedish text.

Usage:
    python scripts/eval_perplexity.py --base LiquidAI/LFM2.5-1.2B-Base --cpt outputs/lfm25-svenska-1.2b-cpt
    python scripts/eval_perplexity.py --base LiquidAI/LFM2.5-1.2B-Base --cpt outputs/lfm25-svenska-1.2b-cpt --sft outputs/lfm25-svenska-1.2b-cpt-sft
"""

import argparse
import json
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False


def load_eval_texts(path):
    """Load eval JSONL manifest."""
    texts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            if text:
                texts.append(text)
    return texts


def compute_perplexity(model, tokenizer, texts, seq_length=2048, device="cpu"):
    """Compute perplexity on a list of texts using sliding window."""
    model.to(device)

    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        input_ids = tokenizer(text, truncation=False, return_tensors="pt")["input_ids"]
        seq_len = input_ids.size(1)

        # Sliding window for long texts
        stride = seq_length
        for begin in range(0, seq_len, stride):
            end = min(begin + seq_length, seq_len)
            chunk = input_ids[:, begin:end].to(device)

            with torch.no_grad():
                outputs = model(chunk, labels=chunk)
                loss = outputs.loss

            num_tokens = chunk.size(1)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    return {"avg_loss": avg_loss, "perplexity": perplexity, "total_tokens": total_tokens}


def load_model(model_path, dtype):
    """Load model and tokenizer, using Unsloth for 2x faster inference when available."""
    if HAS_UNSLOTH:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                load_in_4bit=False,
            )
            FastLanguageModel.for_inference(model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer
        except Exception as e:
            print(f"  Unsloth load failed ({e}), falling back to transformers")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Perplexity evaluation: base vs CPT")
    parser.add_argument("--base", required=True, help="Base model name/path")
    parser.add_argument("--cpt", required=True, help="CPT model name/path")
    parser.add_argument("--sft", default=None, help="SFT model name/path (optional, for 3-way comparison)")
    parser.add_argument("--eval_data", default="data/manifests/cpt_eval.jsonl", help="Eval data manifest")
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_docs", type=int, default=None, help="Limit eval docs")
    parser.add_argument("--output", default="outputs/eval_perplexity.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    # Load eval data
    print("Loading eval data...")
    texts = load_eval_texts(args.eval_data)
    if args.max_docs:
        texts = texts[: args.max_docs]
    print(f"  {len(texts)} eval documents")

    # Evaluate base
    print(f"\nEvaluating base: {args.base}")
    base_model, base_tokenizer = load_model(args.base, dtype)
    base_results = compute_perplexity(base_model, base_tokenizer, texts, args.seq_length, device)
    print(f"  Base perplexity: {base_results['perplexity']:.2f}")
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Evaluate CPT
    print(f"\nEvaluating CPT: {args.cpt}")
    cpt_model, cpt_tokenizer = load_model(args.cpt, dtype)
    cpt_results = compute_perplexity(cpt_model, cpt_tokenizer, texts, args.seq_length, device)
    print(f"  CPT perplexity: {cpt_results['perplexity']:.2f}")
    del cpt_model

    # Evaluate SFT (optional)
    sft_results = None
    if args.sft:
        print(f"\nEvaluating SFT: {args.sft}")
        sft_model, sft_tokenizer = load_model(args.sft, dtype)
        sft_results = compute_perplexity(sft_model, sft_tokenizer, texts, args.seq_length, device)
        print(f"  SFT perplexity: {sft_results['perplexity']:.2f}")
        del sft_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compare
    cpt_delta_pct = ((cpt_results["perplexity"] - base_results["perplexity"]) / base_results["perplexity"]) * 100
    results = {
        "base_model": args.base,
        "cpt_model": args.cpt,
        "eval_docs": len(texts),
        "base_ppl": round(base_results["perplexity"], 4),
        "cpt_ppl": round(cpt_results["perplexity"], 4),
        "cpt_delta_pct": round(cpt_delta_pct, 2),
        "base_loss": round(base_results["avg_loss"], 4),
        "cpt_loss": round(cpt_results["avg_loss"], 4),
    }

    if sft_results:
        sft_delta_pct = ((sft_results["perplexity"] - base_results["perplexity"]) / base_results["perplexity"]) * 100
        results["sft_model"] = args.sft
        results["sft_ppl"] = round(sft_results["perplexity"], 4)
        results["sft_delta_pct"] = round(sft_delta_pct, 2)
        results["sft_loss"] = round(sft_results["avg_loss"], 4)

    print(f"\n=== Perplexity Comparison ===")
    print(f"Base: {results['base_ppl']}")
    print(f"CPT:  {results['cpt_ppl']} ({results['cpt_delta_pct']:+.2f}%)")
    if cpt_delta_pct < 0:
        print("  => CPT IMPROVED perplexity on Swedish eval data")
    else:
        print("  => WARNING: CPT did NOT improve perplexity")
    if sft_results:
        print(f"SFT:  {results['sft_ppl']} ({results['sft_delta_pct']:+.2f}%)")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
