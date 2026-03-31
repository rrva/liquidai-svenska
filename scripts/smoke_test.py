"""Smoke test: load LFM2.5-1.2B-Base, tokenize Swedish text, run one forward pass."""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Base"

SWEDISH_SAMPLES = [
    "Sverige är ett land i Skandinavien med en befolkning på cirka tio miljoner människor.",
    "Den svenska välfärdsmodellen har länge betraktats som en förebild för andra länder.",
    "Stockholm, Göteborg och Malmö är Sveriges tre största städer.",
    "Allemansrätten ger alla rätt att röra sig fritt i naturen, oavsett vem som äger marken.",
    "Fika är en viktig del av svensk kultur och innebär en paus för kaffe och något att äta.",
]


def main():
    print(f"=== Smoke test: {MODEL_NAME} ===\n")

    # 1. Load tokenizer
    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Model max length: {tokenizer.model_max_length}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   Set pad_token = eos_token ({tokenizer.eos_token!r})")

    # 2. Tokenize Swedish text
    print("\n2. Tokenizing Swedish samples...")
    for i, text in enumerate(SWEDISH_SAMPLES):
        token_ids = tokenizer(text)["input_ids"]
        decoded = tokenizer.decode(token_ids)
        print(f"   [{i}] {len(token_ids)} tokens | first 10: {token_ids[:10]}")
        roundtrip_ok = decoded.strip() == text.strip()
        if not roundtrip_ok:
            print(f"   WARNING: roundtrip mismatch")
            print(f"     original: {text[:80]}")
            print(f"     decoded:  {decoded[:80]}")

    # 3. Batch tokenize
    print("\n3. Batch tokenization...")
    batch = tokenizer(
        SWEDISH_SAMPLES,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    )
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   attention_mask shape: {batch['attention_mask'].shape}")

    # 4. Load model
    print("\n4. Loading model...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    print(f"   Device: {device}, dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,} ({param_count/1e9:.2f}B)")

    # 5. Single forward pass
    print("\n5. Forward pass...")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    loss = outputs.loss
    logits_shape = outputs.logits.shape
    print(f"   Loss: {loss}")
    print(f"   Logits shape: {logits_shape}")

    # 6. Short generation test
    print("\n6. Generation test...")
    prompt = "Sverige är känt för"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        gen = model.generate(input_ids, max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9)
    generated_text = tokenizer.decode(gen[0], skip_special_tokens=True)
    print(f"   Prompt: {prompt!r}")
    print(f"   Generated: {generated_text!r}")

    print("\n=== Smoke test PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
