"""
Chat evaluation: run matched Swedish prompts on base, CPT, SFT, and SFT-only models.

Usage:
    python scripts/eval_chat.py --base LiquidAI/LFM2.5-1.2B-Base --cpt outputs/lfm25-svenska-1.2b-cpt
    python scripts/eval_chat.py --base LiquidAI/LFM2.5-1.2B-Base --cpt outputs/lfm25-svenska-1.2b-cpt --sft outputs/lfm25-svenska-1.2b-cpt-sft
    python scripts/eval_chat.py --base LiquidAI/LFM2.5-1.2B-Base --cpt outputs/lfm25-svenska-1.2b-cpt --sft outputs/lfm25-svenska-1.2b-cpt-sft --sft_only outputs/lfm25-svenska-1.2b-sft-only
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Swedish eval prompts covering key areas
DEFAULT_PROMPTS = [
    # Particles (ju, väl, nog)
    "Det är ju så att",
    "Du kommer väl på festen",
    "Det räcker nog med",
    # Compound words
    "Sveriges arbetsmarknadspolitik handlar om",
    "Barnomsorgen i Stockholm",
    # Formal vs casual
    "Jag vill härmed informera er om att",
    "Tjena! Vad händer ikväll",
    # Idiomatic / cultural
    "Lagom är bäst, det betyder att",
    "I Sverige firar man midsommar genom att",
    # Practical / everyday
    "För att ansöka om personnummer i Sverige behöver du",
    "BankID används i Sverige för att",
    # Multi-turn seed
    "Hej, jag har en fråga om",
    # News-style
    "Enligt nya uppgifter från Statistiska centralbyrån",
    # Nature / geography
    "Norrland kännetecknas av",
    # Food
    "Klassiska svenska maträtter inkluderar",
]


def load_prompts(path):
    """Load prompts from file (one per line), skip empty/comments."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def _is_peft_adapter(model_path):
    """Check if a model path contains a PEFT adapter (not a full model)."""
    return Path(model_path).is_dir() and (Path(model_path) / "adapter_config.json").exists()


def load_model(model_path, dtype, device, base_model_path=None):
    """Load model and tokenizer, auto-detecting PEFT adapters."""
    if _is_peft_adapter(model_path):
        from peft import PeftModel

        if base_model_path is None:
            with open(Path(model_path) / "adapter_config.json") as f:
                adapter_cfg = json.load(f)
            base_model_path = adapter_cfg.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError(
                f"{model_path} is a PEFT adapter but no base model could be determined. "
                "Pass the base model path via --cpt (for SFT adapters) or check adapter_config.json."
            )
        print(f"  Loading PEFT adapter (base: {base_model_path})")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=dtype, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        )
    model.to(device)
    return model, tokenizer


def generate_responses(model, tokenizer, prompts, device, max_new_tokens=100):
    """Generate text for each prompt using greedy decoding for deterministic results."""
    responses = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
            )
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract only the generated continuation
        generated = full_text[len(prompt):].strip()
        responses.append(generated)
    return responses


def format_comparison(prompts, base_responses, cpt_responses, sft_responses=None, sft_only_responses=None):
    """Format as markdown comparison."""
    parts = ["Base", "CPT"]
    if sft_responses:
        parts.append("CPT+SFT")
    if sft_only_responses:
        parts.append("SFT-only")
    title = " vs ".join(parts)
    lines = [f"# Chat Evaluation: {title}\n"]
    for i, prompt in enumerate(prompts):
        lines.append(f"## Prompt {i+1}")
        lines.append(f"**Prompt:** {prompt}\n")
        lines.append(f"**Base:**\n> {base_responses[i]}\n")
        lines.append(f"**CPT:**\n> {cpt_responses[i]}\n")
        if sft_responses:
            lines.append(f"**CPT+SFT:**\n> {sft_responses[i]}\n")
        if sft_only_responses:
            lines.append(f"**SFT-only:**\n> {sft_only_responses[i]}\n")
        lines.append("---\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Chat evaluation: base vs CPT")
    parser.add_argument("--base", required=True, help="Base model name/path")
    parser.add_argument("--cpt", required=True, help="CPT model name/path")
    parser.add_argument("--sft", default=None, help="SFT model name/path (CPT+SFT, optional)")
    parser.add_argument("--sft_only", default=None, help="SFT-only model name/path (SFT on base, no CPT — ablation baseline)")
    parser.add_argument("--prompts", default=None, help="Prompts file (one per line)")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    # Load prompts
    if args.prompts:
        prompts = load_prompts(args.prompts)
    else:
        prompts = DEFAULT_PROMPTS
    print(f"Using {len(prompts)} prompts")

    # Base model
    print(f"\nLoading base: {args.base}")
    base_model, base_tokenizer = load_model(args.base, dtype, device)

    print("Generating base responses...")
    base_responses = generate_responses(base_model, base_tokenizer, prompts, device, args.max_new_tokens)
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # CPT model
    print(f"\nLoading CPT: {args.cpt}")
    cpt_model, cpt_tokenizer = load_model(args.cpt, dtype, device)

    print("Generating CPT responses...")
    cpt_responses = generate_responses(cpt_model, cpt_tokenizer, prompts, device, args.max_new_tokens)
    del cpt_model

    # SFT model (optional, auto-detects PEFT adapters)
    sft_responses = None
    if args.sft:
        print(f"\nLoading SFT: {args.sft}")
        sft_model, sft_tokenizer = load_model(args.sft, dtype, device, base_model_path=args.cpt)

        print("Generating SFT responses...")
        sft_responses = generate_responses(sft_model, sft_tokenizer, prompts, device, args.max_new_tokens)
        del sft_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # SFT-only baseline (optional — SFT directly on base, no CPT)
    sft_only_responses = None
    if args.sft_only:
        print(f"\nLoading SFT-only: {args.sft_only}")
        sft_only_model, sft_only_tokenizer = load_model(args.sft_only, dtype, device, base_model_path=args.base)

        print("Generating SFT-only responses...")
        sft_only_responses = generate_responses(sft_only_model, sft_only_tokenizer, prompts, device, args.max_new_tokens)
        del sft_only_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save individual outputs
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    comparison_md = format_comparison(prompts, base_responses, cpt_responses, sft_responses, sft_only_responses)

    (out / "eval_chat_base.md").write_text(
        "# Base Model Responses\n\n" +
        "\n\n".join(f"**{p}**\n> {r}" for p, r in zip(prompts, base_responses))
    )
    (out / "eval_chat_cpt.md").write_text(
        "# CPT Model Responses\n\n" +
        "\n\n".join(f"**{p}**\n> {r}" for p, r in zip(prompts, cpt_responses))
    )
    if sft_responses:
        (out / "eval_chat_sft.md").write_text(
            "# SFT Model Responses\n\n" +
            "\n\n".join(f"**{p}**\n> {r}" for p, r in zip(prompts, sft_responses))
        )
    if sft_only_responses:
        (out / "eval_chat_sft_only.md").write_text(
            "# SFT-only Model Responses (no CPT)\n\n" +
            "\n\n".join(f"**{p}**\n> {r}" for p, r in zip(prompts, sft_only_responses))
        )
    (out / "eval_chat_comparison.md").write_text(comparison_md)

    print(f"\nSaved to {out}/eval_chat_*.md")
    print("\n=== Sample Comparisons ===")
    for i in range(min(3, len(prompts))):
        print(f"\nPrompt: {prompts[i]}")
        print(f"Base:     {base_responses[i][:120]}...")
        print(f"CPT:      {cpt_responses[i][:120]}...")
        if sft_responses:
            print(f"CPT+SFT:  {sft_responses[i][:120]}...")
        if sft_only_responses:
            print(f"SFT-only: {sft_only_responses[i][:120]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
