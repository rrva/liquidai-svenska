# Swedish LFM2.5 Execution Plan (CPT → SFT)

## Mission

Build a Swedish-adapted conversational model in two stages:
1. **Continued pretraining (CPT)** on Swedish raw text using the base checkpoint
2. **Supervised fine-tuning (SFT)** on Swedish instruction/chat data using the CPT checkpoint

- Base model: `LiquidAI/LFM2.5-1.2B-Base`
- Chat target: Swedish conversational assistant
- Infra: Hugging Face Jobs with UV scripts (Unsloth Jobs)
- Local workflow: repo-first, reproducible, resumable

### Unsloth Jobs reference (from HF blog)
- Official LFM2.5 SFT template: `https://huggingface.co/datasets/unsloth/jobs/resolve/main/sft-lfm2.5.py`
- GPU flavors: `t4-small` ~$0.40/hr, `t4-medium` ~$0.60/hr, `a10g-small` ~$1.00/hr, `a10g-large` ~$3.00/hr
- Free credits: join the `unsloth-jobs` HF org
- Monitoring: use `report_to="trackio"` for real-time loss curves
- Claude Code plugin: `huggingface/skills` marketplace (optional)
- CPT note: Unsloth Jobs templates are SFT-oriented (LoRA). CPT (full-parameter) needs a custom UV script using plain `transformers` + `Trainer`, not Unsloth's `FastLanguageModel`

### Non-goals
- Do not start with 2.6B
- Do not build a fancy eval dashboard first
- Do not generate GGUF during the first CPT run
- Do not over-optimize hyperparameters before we have a baseline

### Success criteria

**Phase A — CPT**: visibly better Swedish fluency, lower perplexity on held-out Swedish, fewer English intrusions.

**Phase B — SFT**: stable Swedish chat, better instruction following than CPT-only, better tone/helpfulness/multi-turn than stock instruct for Swedish.

---

## Repo layout

```
liquidai-svenska/
├── README.md
├── .gitignore
├── Makefile
├── configs/
│   ├── cpt_1.2b.yaml
│   ├── sft_1.2b.yaml
│   └── datasets.yaml
├── scripts/
│   ├── prepare_cpt_data.py
│   ├── prepare_sft_data.py
│   ├── train_cpt_hfjobs.py
│   ├── train_sft_hfjobs.py
│   ├── eval_perplexity.py
│   ├── eval_chat.py
│   └── smoke_test.py
├── data/
│   ├── README.md
│   ├── manifests/
│   └── samples/
├── prompts/
│   ├── eval_prompts_sv.txt
│   └── style_prompts_sv.txt
└── outputs/
    └── .gitkeep
```

---

## Defaults

| Setting | Value |
|---------|-------|
| CPT input model | `LiquidAI/LFM2.5-1.2B-Base` |
| SFT input model | CPT output checkpoint |
| CPT seq length | 2048 |
| SFT seq length | 4096 |
| Precision | bf16 (fp16 fallback only if required) |
| CPT pilot tokens | 100M–250M |
| CPT meaningful run | 500M–1B |
| SFT examples | 50K–200K |

---

## High-level strategy

**CPT first** — Swedish fluency, morphology, compounds, common function words, reducing English-first latent priors.

**SFT second** — instruction following, tone, refusal style, dialog rhythm, real assistant behavior.

**Trade-off**: CPT without enough clean tokens is easy to waste. SFT without enough conversational variety yields robotic Swedish. Practical compromise: small disciplined CPT, then larger better SFT.

---

## Data plan

### 1) CPT data

**Target token mix:**
- 40% encyclopedic / factual prose
- 20% news / journalistic
- 20% books / essays / long-form prose (if licensing allows)
- 10% forums / conversational Swedish
- 10% public-domain / government / documentation / practical prose

**Hard constraints — exclude:**
- Machine-translated junk
- OCR trash
- Duplicated mirrors
- Code-heavy files
- Mixed-language junk
- Short snippets with no discourse value
- Legally dubious scraped sources

**Quality pipeline:**
1. Detect language, keep only Swedish-dominant
2. Normalize unicode
3. Strip boilerplate
4. Deduplicate aggressively
5. Filter very short / low-information lines
6. Preserve paragraph boundaries
7. Pack into long training sequences later

**Deliverables:**
- `data/manifests/cpt_sources.jsonl`
- `data/manifests/cpt_train.jsonl`
- `data/manifests/cpt_eval.jsonl`

Record format:
```json
{"id": "source-doc-000001", "source": "example_source_name", "license": "license-or-unknown", "text": "full cleaned swedish text ...", "chars": 12345}
```

Split: 99% train / 1% eval, deduplicate before split, group by source to avoid leakage.

### 2) SFT data

**Initial mix:**
- 50% instruction / QA
- 30% natural dialog
- 20% multi-turn synthetic conversations

**Known candidates** (must verify live before implementation):
- `neph1/Alpaca-Lora-GPT4-Swedish-Refined` (52K, cleaned Swedish Alpaca)
- `elliottdury/svenska-wikipedia-qa-small` (100K+ QA)
- `OpenAssistant/oasst2` filtered `lang=="sv"` (human multi-turn)
- `alexandrainst/scandi-qa` sv split (6.8K)
- `skvarre/swedish-instruct-data-chatgpt4` (GPT-4 generated)

**Chat format:**
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Deliverables:** `data/manifests/sft_train.jsonl`, `data/manifests/sft_eval.jsonl` (95/5 split, stratified by source).

### 3) Synthetic data for SFT

Target: 50K–100K synthetic Swedish conversations.

**Categories:** casual chat, practical advice, customer support, small business admin, email drafting, SMS replies, planning, parenting/school, polite refusals, clarifications, short multi-turn followups, factual explanations.

**Style axes:** formal/neutral/casual, concise/explanatory, warm/dry/practical, text message vs email vs spoken chat.

**Quality rules — reject outputs that:** sound translated, overuse English calques, feel too American culturally, over-explain, have broken Swedish word order, contain policy boilerplate.

**Workflow:** prefer direct Swedish generation over translation, run quality filter pass, dedupe semantically, keep provenance tags.

---

## Implementation phases

### Phase 0 — Scaffold and sanity checks

- Create repo structure
- Add .gitignore, README.md, YAML configs
- Add `smoke_test.py` that: loads tokenizer, tokenizes Swedish text, loads 1-batch dummy dataset, runs single forward pass

**Gate 1**: Stop if model doesn't load, tokenizer is broken, or sequence packing is unstable.

### Phase 1 — CPT data pipeline

`scripts/prepare_cpt_data.py`

CLI: `python scripts/prepare_cpt_data.py --config configs/datasets.yaml --out data/manifests --min_chars 200 --dedupe`

Features: source-level metadata, exact/near dedupe, language filtering, text length filtering, train/eval split, sample preview output.

Output stats: num docs, total chars, estimated tokens, source distribution, average doc length, duplicate rates.

### Phase 2 — CPT training job

`scripts/train_cpt_hfjobs.py` — self-contained UV script.

**This is NOT LoRA. CPT updates full model weights.** Cannot use Unsloth's `FastLanguageModel` (LoRA-oriented). Use plain `transformers.Trainer` with causal LM objective.

```python
# /// script
# dependencies = ["transformers>=4.45.0", "datasets", "accelerate", "trackio", "torch"]
# ///
```

CPT config v1:
```yaml
model_name: LiquidAI/LFM2.5-1.2B-Base
seq_length: 2048
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
weight_decay: 0.1
warmup_ratio: 0.01
lr_scheduler_type: cosine
num_train_epochs: 1
bf16: true
gradient_checkpointing: true
logging_steps: 10
save_steps: 500
eval_steps: 500
save_total_limit: 3
```

Run (use exact flavor name available in your account at execution time):
```bash
hf jobs uv run scripts/train_cpt_hfjobs.py \
    --flavor a100-small \
    --secrets HF_TOKEN \
    --timeout 24h \
    -- --config configs/cpt_1.2b.yaml
```

Cost estimate: full-param 1.2B on A100 for ~500M tokens ≈ a few hours ≈ $5–15. Pilot (100M tokens) ≈ $2–5.

Output repo: `<USERNAME>/lfm25-svenska-1.2b-cpt`

Save: tokenizer, config snapshot, trainer state, final checkpoint, run_summary.json.

If full-param CPT too expensive, fall back to QLoRA-style domain adaptation as compromise experiment only.

**Gate 2**: Proceed only if loss decreases, Swedish perplexity improves, generation still coherent.

### Phase 3 — CPT evaluation

`scripts/eval_perplexity.py` — compare base vs CPT on held-out Swedish.
Output: `{"base_ppl": ..., "cpt_ppl": ..., "delta_pct": ...}`

`scripts/eval_chat.py` — run matched prompts on base and CPT.
Test: particles (ju, väl, nog), compounds, formal vs casual, idiomatic chat, culturally Swedish phrasing.
Store: `outputs/eval_chat_base.md`, `outputs/eval_chat_cpt.md`

### Phase 4 — SFT data pipeline

`scripts/prepare_sft_data.py`

CLI: `python scripts/prepare_sft_data.py --config configs/datasets.yaml --out data/manifests --synthetic-target 50000`

Must inspect every dataset schema before coding adapters. Keep dataset-specific adapters in clean functions. Validate all rows. Remove low-quality/empty outputs. Keep provenance.

### Phase 5 — SFT training

`scripts/train_sft_hfjobs.py` — train on CPT checkpoint with LoRA.

SFT config v1:
```yaml
model_name: <USERNAME>/lfm25-svenska-1.2b-cpt
seq_length: 4096
load_in_4bit: false
lora_r: 16
lora_alpha: 16
lora_dropout: 0.05
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.01
bf16: true
gradient_checkpointing: true
logging_steps: 10
save_steps: 250
eval_steps: 250
```

If Unsloth supports the CPT checkpoint cleanly, use it for SFT (base `train_sft_hfjobs.py` on the official Unsloth LFM2.5 SFT template at `https://huggingface.co/datasets/unsloth/jobs/resolve/main/sft-lfm2.5.py`). If not, fall back to transformers + trl. Do not block on framework purity. Use `report_to="trackio"` for monitoring.

Output repo: `<USERNAME>/lfm25-svenska-1.2b-cpt-sft`

**Gate 3**: Proceed to polish only if Swedish chat visibly improved, not overfit/repetitive, no glaring regressions.

### Phase 6 — Final evaluation

Compare: Base → CPT → CPT+SFT

**Automatic:** perplexity, QA match, response length stability, refusal consistency.

**Manual (50+ prompts):** idiomatic Swedish, chat naturalness, instruction following, office/email/SMS tone, cultural grounding, English intrusion rate.

Categories: chat, practical help, factual QA, Swedish workplace, parent/teacher, casual texts, messy spoken Swedish.

Store: `outputs/final_eval.md`

---

## Makefile targets

```makefile
smoke:
	python scripts/smoke_test.py
prepare-cpt:
	python scripts/prepare_cpt_data.py --config configs/datasets.yaml --out data/manifests
prepare-sft:
	python scripts/prepare_sft_data.py --config configs/datasets.yaml --out data/manifests
eval-ppl:
	python scripts/eval_perplexity.py
eval-chat:
	python scripts/eval_chat.py
```

---

## Execution order

1. Scaffold repo
2. Write configs
3. Implement `smoke_test.py`
4. Implement `prepare_cpt_data.py`
5. Create tiny local CPT sample corpus
6. Run local tokenizer/packing smoke test
7. Implement `train_cpt_hfjobs.py`
8. Submit small CPT pilot
9. Evaluate CPT
10. Implement `prepare_sft_data.py`
11. Implement `train_sft_hfjobs.py`
12. Run SFT on CPT checkpoint
13. Run final eval
14. Update README with exact commands that worked

---

## Rules for implementation

- Prefer boring working code over elegant abstractions
- Verify every remote dataset schema before coding adapters
- Log everything important
- Save artifacts often
- Do not hide failures
- If a framework path fails, fall back to plain transformers
- Keep scripts runnable both locally and in HF Jobs where practical

---

## README requirements

Must contain: project goal, model strategy (CPT then SFT), data provenance approach, license caveat section, local quickstart, HF Jobs quickstart, evaluation workflow, known risks.

**License warning**: Do not train on datasets unless licensing is acceptable for model training and redistribution. Do not assume scraped Swedish text is safe. Keep source manifest. Keep per-source allowlist/denylist. Document what went into every training run.

---

## Key facts verified

- `LiquidAI/LFM2.5-1.2B-Base` exists publicly on HF, described as pre-trained checkpoint for heavy fine-tuning
- `LiquidAI/LFM2.5-1.2B-Instruct` exists as instruction-tuned sibling
- HF Jobs supports UV scripts via `hf jobs uv run`
- Unsloth Jobs blog (https://huggingface.co/blog/unsloth-jobs) confirms LFM2.5 support with example SFT script
- GPU flavors verified: t4-small, t4-medium, a10g-small, a10g-large; A100 flavors for CPT
- Free credits available via `unsloth-jobs` HF org
- 2.6B not attempted until 1.2B pipeline is proven
