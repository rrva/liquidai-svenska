"""Shared model loading utilities for eval scripts."""

import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def is_peft_adapter(model_path):
    """Check if a model path contains a PEFT adapter (not a full model)."""
    return Path(model_path).is_dir() and (Path(model_path) / "adapter_config.json").exists()


def load_model(model_path, dtype, device=None, base_model_path=None):
    """Load model and tokenizer, auto-detecting PEFT adapters.

    Args:
        model_path: Path or HF repo ID for the model.
        dtype: Torch dtype (e.g. torch.bfloat16).
        device: Device to move model to (optional, skipped if None).
        base_model_path: Explicit base model for PEFT adapters.

    Returns:
        (model, tokenizer) tuple.
    """
    if is_peft_adapter(model_path):
        from peft import PeftModel

        if base_model_path is None:
            with open(Path(model_path) / "adapter_config.json") as f:
                adapter_cfg = json.load(f)
            base_model_path = adapter_cfg.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError(
                f"{model_path} is a PEFT adapter but no base model could be determined. "
                "Pass the base model path explicitly or check adapter_config.json."
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

    if device is not None:
        model.to(device)

    return model, tokenizer
