# Swedish LFM2.5 (liquidai-svenska)

Swedish-adapted conversational model built in two stages on top of [LiquidAI/LFM2.5-1.2B-Base](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base):

1. **Continued Pretraining (CPT)** — full-parameter training on clean Swedish text
2. **Supervised Fine-Tuning (SFT)** — LoRA fine-tuning on Swedish instruction/chat data

## Strategy

CPT injects Swedish fluency, morphology, and vocabulary into the base model. SFT adds instruction following, dialog behavior, and tone. The 1.2B model is the starting point; 2.6B only after this pipeline is proven.

## Quick start (local)

```bash
# Smoke test — loads model, tokenizes Swedish, runs forward pass
make smoke

# Prepare CPT data (downloads from HF, needs HF_TOKEN for gated datasets)
export HF_TOKEN=your_token
make prepare-cpt

# Prepare SFT data
make prepare-sft

# Prepare SFT data with synthetic generation (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=your_key
python scripts/prepare_sft_data.py --config configs/datasets.yaml --out data/manifests --synthetic_target 50000
```

## HF Jobs (training)

### CPT (full-parameter)

```bash
hf jobs uv run scripts/train_cpt_hfjobs.py \
    --flavor a100-small --secrets HF_TOKEN --timeout 24h \
    -- --config configs/cpt_1.2b.yaml
```

### SFT (LoRA via Unsloth)

```bash
hf jobs uv run scripts/train_sft_hfjobs.py \
    --flavor a10g-small --secrets HF_TOKEN --timeout 6h \
    -- --config configs/sft_1.2b.yaml
```

If Unsloth fails to load the CPT checkpoint, add `--no_unsloth` to fall back to transformers+trl.

## Evaluation

```bash
# Perplexity comparison
python scripts/eval_perplexity.py \
    --base LiquidAI/LFM2.5-1.2B-Base \
    --cpt outputs/lfm25-svenska-1.2b-cpt

# Chat comparison (side-by-side generation)
python scripts/eval_chat.py \
    --base LiquidAI/LFM2.5-1.2B-Base \
    --cpt outputs/lfm25-svenska-1.2b-cpt \
    --prompts prompts/eval_prompts_sv.txt
```

## Data provenance

All training data is tracked in JSONL manifests under `data/manifests/`. Each document records its source and license field. See `configs/datasets.yaml` for the full source list.

### CPT sources
- Swedish Wikipedia (CC-BY-SA-3.0) — encyclopedic prose
- CulturaX Swedish subset (various, gated) — web-crawled text
- MC4 Swedish (ODC-BY-1.0) — cleaned Common Crawl / news
- OSCAR Swedish (CC0-1.0) — web corpus / forums
- Swedish Riksdag (CC0-1.0) — government / parliamentary text
- Swedish literature (CC-BY-4.0) — books / long-form prose

### SFT sources
- `neph1/Alpaca-Lora-GPT4-Swedish-Refined` (52K instruction pairs)
- `elliottdury/svenska-wikipedia-qa-small` (100K QA pairs)
- `OpenAssistant/oasst2` Swedish subset (multi-turn dialog)
- `alexandrainst/scandi-qa` Swedish split (6.8K QA)
- `skvarre/swedish-instruct-data-chatgpt4` (1.4K instruction pairs)
- Synthetic Swedish conversations (generated via Anthropic API, `--synthetic_target`)

## License warning

Not all upstream datasets have clear redistribution licenses. Before publishing any model trained with this pipeline:

- Verify each dataset's license permits model training and redistribution
- Check the `license` field in manifest files — each record carries its source license
- Entries marked `check-upstream` or `unknown` require manual verification
- Entries marked `synthetic` were generated via API and have no upstream license concern
- Do not assume scraped Swedish text is safe to use

## Repo structure

```
configs/          — Training and dataset YAML configs
scripts/          — All executable scripts (data prep, training, eval)
data/manifests/   — JSONL data manifests (tracked)
data/samples/     — Small test samples (tracked)
prompts/          — Swedish eval and style prompts
outputs/          — Training outputs and eval results (gitignored)
```

## Known risks

- CPT with too few tokens may not meaningfully improve Swedish fluency
- SFT data quality varies; some datasets may contain machine-translated text
- LFM2.5 architecture uses custom modules — not all frameworks support it equally
- Unsloth compatibility with CPT checkpoints is unverified until first run
