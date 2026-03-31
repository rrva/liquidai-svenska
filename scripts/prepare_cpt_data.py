"""
CPT data pipeline: download, clean, deduplicate, split Swedish text for continued pretraining.

Usage:
    python scripts/prepare_cpt_data.py --config configs/datasets.yaml --out data/manifests
    python scripts/prepare_cpt_data.py --config configs/datasets.yaml --out data/manifests --min_chars 200 --dedupe
"""

import argparse
import hashlib
import json
import os
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def normalize_text(text):
    """Unicode normalize, strip boilerplate whitespace."""
    text = unicodedata.normalize("NFC", text)
    # Collapse runs of whitespace but preserve paragraph breaks
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned.append(line)
        elif cleaned and cleaned[-1] != "":
            cleaned.append("")
    return "\n".join(cleaned).strip()


def text_hash(text):
    """SHA-256 hash for exact dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def passes_quality(text, min_chars, max_chars, min_words):
    """Basic quality filters."""
    if len(text) < min_chars:
        return False
    if len(text) > max_chars:
        return False
    words = text.split()
    if len(words) < min_words:
        return False
    # Reject if mostly non-alphabetic (likely junk)
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars / max(len(text), 1) < 0.5:
        return False
    return True


def load_hf_source(source_cfg, token=None):
    """Load documents from a HF dataset source config."""
    from datasets import load_dataset

    path = source_cfg["path"]
    subset = source_cfg.get("subset")
    split = source_cfg.get("split", "train")
    text_field = source_cfg.get("text_field", "text")
    lang_filter = source_cfg.get("lang_filter")
    max_docs = source_cfg.get("max_docs")
    name = source_cfg["name"]

    print(f"  Loading {name}: {path} subset={subset} split={split}")

    kwargs = {"split": split, "trust_remote_code": True}
    if subset:
        kwargs["name"] = subset
    if token:
        kwargs["token"] = token

    # Use streaming for large datasets
    if max_docs and max_docs > 0:
        kwargs["streaming"] = True
        ds = load_dataset(path, **kwargs)
        docs = []
        for i, row in enumerate(ds):
            if i >= max_docs:
                break
            text = row.get(text_field, "")
            if lang_filter:
                row_lang = row.get("lang") or row.get("language") or row.get("locale", "")
                if not row_lang.startswith(lang_filter):
                    continue
            if text:
                docs.append({"text": text, "source": name})
        return docs
    else:
        ds = load_dataset(path, **kwargs)
        docs = []
        for row in ds:
            text = row.get(text_field, "")
            if lang_filter:
                row_lang = row.get("lang") or row.get("language") or row.get("locale", "")
                if not row_lang.startswith(lang_filter):
                    continue
            if text:
                docs.append({"text": text, "source": name})
        return docs


def load_local_source(source_cfg):
    """Load documents from local JSONL files (e.g., samples)."""
    path = source_cfg["path"]
    name = source_cfg["name"]
    text_field = source_cfg.get("text_field", "text")
    docs = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            text = row.get(text_field, "")
            if text:
                docs.append({"text": text, "source": name})
    return docs


def main():
    parser = argparse.ArgumentParser(description="Prepare CPT data for Swedish LFM2.5")
    parser.add_argument("--config", required=True, help="Path to datasets.yaml")
    parser.add_argument("--out", required=True, help="Output directory for manifests")
    parser.add_argument("--min_chars", type=int, default=200)
    parser.add_argument("--max_chars", type=int, default=100000)
    parser.add_argument("--min_words", type=int, default=30)
    parser.add_argument("--no_dedupe", action="store_true", help="Disable deduplication")
    parser.add_argument("--eval_ratio", type=float, default=None, help="Override eval split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_only", action="store_true", help="Skip HF downloads, use only local sources")
    args = parser.parse_args()

    random.seed(args.seed)
    config = load_config(args.config)
    quality_cfg = config.get("quality", {})
    min_chars = args.min_chars or quality_cfg.get("min_chars", 200)
    max_chars = args.max_chars or quality_cfg.get("max_chars", 100000)
    min_words = args.min_words or quality_cfg.get("min_words", 30)
    eval_ratio = args.eval_ratio or config.get("splits", {}).get("cpt_eval_ratio", 0.01)

    token = os.environ.get("HF_TOKEN")
    if not token and not args.local_only:
        print("WARNING: HF_TOKEN not set. Gated datasets (e.g., CulturaX) will fail.")

    # Collect all documents
    all_docs = []
    sources = config.get("cpt_sources", [])
    for src in sources:
        src_type = src.get("type", "huggingface")
        try:
            if args.local_only and src_type == "huggingface":
                print(f"  Skipping HF source: {src['name']} (--local_only)")
                continue
            if src_type == "huggingface":
                docs = load_hf_source(src, token=token)
            elif src_type == "local":
                docs = load_local_source(src)
            else:
                print(f"  Unknown source type: {src_type}, skipping {src['name']}")
                continue
            print(f"  -> {len(docs)} raw docs from {src['name']}")
            all_docs.extend(docs)
        except Exception as e:
            print(f"  ERROR loading {src['name']}: {e}")
            continue

    print(f"\nTotal raw docs: {len(all_docs)}")

    # Normalize
    print("Normalizing text...")
    for doc in all_docs:
        doc["text"] = normalize_text(doc["text"])

    # Quality filter
    print("Quality filtering...")
    before = len(all_docs)
    all_docs = [d for d in all_docs if passes_quality(d["text"], min_chars, max_chars, min_words)]
    print(f"  {before} -> {len(all_docs)} after quality filter")

    # Dedup
    if not args.no_dedupe:
        print("Deduplicating...")
        seen = set()
        deduped = []
        for doc in all_docs:
            h = text_hash(doc["text"])
            if h not in seen:
                seen.add(h)
                deduped.append(doc)
        print(f"  {len(all_docs)} -> {len(deduped)} after dedup ({len(all_docs) - len(deduped)} duplicates)")
        all_docs = deduped

    # Assign IDs and compute stats
    for i, doc in enumerate(all_docs):
        doc["id"] = f"cpt-{doc['source']}-{i:07d}"
        doc["chars"] = len(doc["text"])
        doc["license"] = "see-source"

    # Split by source to avoid leakage, then sample eval
    print(f"Splitting train/eval (eval_ratio={eval_ratio})...")
    random.shuffle(all_docs)
    n_eval = max(1, int(len(all_docs) * eval_ratio))
    eval_docs = all_docs[:n_eval]
    train_docs = all_docs[n_eval:]

    # Write manifests
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "cpt_train.jsonl"
    eval_path = out_dir / "cpt_eval.jsonl"
    sources_path = out_dir / "cpt_sources.jsonl"

    def write_jsonl(path, docs):
        with open(path, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    write_jsonl(train_path, train_docs)
    write_jsonl(eval_path, eval_docs)

    # Source summary
    source_counts = Counter(d["source"] for d in all_docs)
    source_chars = Counter()
    for d in all_docs:
        source_chars[d["source"]] += d["chars"]
    source_summary = []
    for src_name, count in source_counts.most_common():
        source_summary.append({
            "source": src_name,
            "docs": count,
            "total_chars": source_chars[src_name],
            "est_tokens": source_chars[src_name] // 4,  # rough estimate
        })
    write_jsonl(sources_path, source_summary)

    # Print stats
    total_chars = sum(d["chars"] for d in all_docs)
    print(f"\n=== CPT Data Stats ===")
    print(f"Total docs:   {len(all_docs):,}")
    print(f"Train docs:   {len(train_docs):,}")
    print(f"Eval docs:    {len(eval_docs):,}")
    print(f"Total chars:  {total_chars:,}")
    print(f"Est. tokens:  {total_chars // 4:,}")
    print(f"\nSource distribution:")
    for s in source_summary:
        print(f"  {s['source']}: {s['docs']:,} docs, {s['total_chars']:,} chars, ~{s['est_tokens']:,} tokens")
    print(f"\nManifests written to: {out_dir}")
    print(f"  {train_path}")
    print(f"  {eval_path}")
    print(f"  {sources_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
