"""
Chat evaluation: run matched Swedish prompts on base, CPT, and optionally SFT models.

Usage:
    python scripts/eval_chat.py --base LiquidAI/LFM2.5-1.2B-Base --cpt outputs/lfm25-svenska-1.2b-cpt
    python scripts/eval_chat.py --base LiquidAI/LFM2.5-1.2B-Base --cpt outputs/lfm25-svenska-1.2b-cpt --sft outputs/lfm25-svenska-1.2b-cpt-sft
"""

import argparse
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


def generate_responses(model, tokenizer, prompts, device, max_new_tokens=100):
    """Generate text for each prompt."""
    responses = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract only the generated continuation
        generated = full_text[len(prompt):].strip()
        responses.append(generated)
    return responses


def format_comparison(prompts, base_responses, cpt_responses, sft_responses=None):
    """Format as markdown comparison."""
    title = "Base vs CPT vs SFT" if sft_responses else "Base vs CPT"
    lines = [f"# Chat Evaluation: {title}\n"]
    for i, prompt in enumerate(prompts):
        lines.append(f"## Prompt {i+1}")
        lines.append(f"**Prompt:** {prompt}\n")
        lines.append(f"**Base:**\n> {base_responses[i]}\n")
        lines.append(f"**CPT:**\n> {cpt_responses[i]}\n")
        if sft_responses:
            lines.append(f"**SFT:**\n> {sft_responses[i]}\n")
        lines.append("---\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Chat evaluation: base vs CPT")
    parser.add_argument("--base", required=True, help="Base model name/path")
    parser.add_argument("--cpt", required=True, help="CPT model name/path")
    parser.add_argument("--sft", default=None, help="SFT model name/path (optional, for 3-way comparison)")
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
    base_tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    print("Generating base responses...")
    base_responses = generate_responses(base_model, base_tokenizer, prompts, device, args.max_new_tokens)
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # CPT model
    print(f"\nLoading CPT: {args.cpt}")
    cpt_tokenizer = AutoTokenizer.from_pretrained(args.cpt, trust_remote_code=True)
    if cpt_tokenizer.pad_token is None:
        cpt_tokenizer.pad_token = cpt_tokenizer.eos_token
    cpt_model = AutoModelForCausalLM.from_pretrained(
        args.cpt, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    print("Generating CPT responses...")
    cpt_responses = generate_responses(cpt_model, cpt_tokenizer, prompts, device, args.max_new_tokens)
    del cpt_model

    # SFT model (optional)
    sft_responses = None
    if args.sft:
        print(f"\nLoading SFT: {args.sft}")
        sft_tokenizer = AutoTokenizer.from_pretrained(args.sft, trust_remote_code=True)
        if sft_tokenizer.pad_token is None:
            sft_tokenizer.pad_token = sft_tokenizer.eos_token
        sft_model = AutoModelForCausalLM.from_pretrained(
            args.sft, torch_dtype=dtype, trust_remote_code=True
        ).to(device)

        print("Generating SFT responses...")
        sft_responses = generate_responses(sft_model, sft_tokenizer, prompts, device, args.max_new_tokens)
        del sft_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save individual outputs
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    comparison_md = format_comparison(prompts, base_responses, cpt_responses, sft_responses)

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
    (out / "eval_chat_comparison.md").write_text(comparison_md)

    print(f"\nSaved to {out}/eval_chat_*.md")
    print("\n=== Sample Comparisons ===")
    for i in range(min(3, len(prompts))):
        print(f"\nPrompt: {prompts[i]}")
        print(f"Base: {base_responses[i][:120]}...")
        print(f"CPT:  {cpt_responses[i][:120]}...")
        if sft_responses:
            print(f"SFT:  {sft_responses[i][:120]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
