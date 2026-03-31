"""
SFT data pipeline: load, adapt, validate, and split Swedish instruction/chat datasets.

Each HF dataset has its own adapter function based on verified schemas.
Optionally generates synthetic Swedish conversations via the Anthropic API.

Usage:
    python scripts/prepare_sft_data.py --config configs/datasets.yaml --out data/manifests
    python scripts/prepare_sft_data.py --config configs/datasets.yaml --out data/manifests --synthetic_target 50000
"""

import argparse
import hashlib
import json
import os
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def normalize_text(text):
    """Unicode normalize and clean whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text.strip()


def messages_hash(messages):
    """Hash a conversation for dedup."""
    blob = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ─── Dataset-specific adapters ───────────────────────────────────────────────
# Each adapter returns a list of {"messages": [...], "source": str}


def adapt_alpaca_swedish(ds, source_name):
    """
    neph1/Alpaca-Lora-GPT4-Swedish-Refined
    Columns: instruction, input, output, id, original_instruction
    """
    results = []
    for row in ds:
        instruction = normalize_text(row.get("instruction", ""))
        inp = normalize_text(row.get("input", ""))
        output = normalize_text(row.get("output", ""))

        if not instruction or not output:
            continue

        user_content = f"{instruction}\n{inp}".strip() if inp else instruction

        results.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ],
            "source": source_name,
        })
    return results


def adapt_wikipedia_qa_sv(ds, source_name):
    """
    elliottdury/svenska-wikipedia-qa-small
    Columns: user, assistant
    """
    results = []
    for row in ds:
        user = normalize_text(row.get("user", ""))
        assistant = normalize_text(row.get("assistant", ""))

        if not user or not assistant:
            continue
        if len(assistant) < 10:
            continue

        results.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "source": source_name,
        })
    return results


def adapt_oasst2_sv(ds, source_name):
    """
    OpenAssistant/oasst2 — filter lang=="sv", reconstruct conversation trees.
    Columns: message_id, parent_id, text, role (prompter/assistant), lang, message_tree_id
    """
    # Collect all Swedish messages
    sv_messages = {}
    trees = defaultdict(list)
    for row in ds:
        if row.get("lang") != "sv":
            continue
        if row.get("deleted"):
            continue
        msg_id = row["message_id"]
        sv_messages[msg_id] = {
            "message_id": msg_id,
            "parent_id": row.get("parent_id"),
            "text": normalize_text(row.get("text", "")),
            "role": "user" if row.get("role") == "prompter" else "assistant",
            "tree_id": row.get("message_tree_id"),
            "rank": row.get("rank"),
        }
        trees[row.get("message_tree_id")].append(msg_id)

    # Reconstruct conversations: follow parent chains from leaf to root
    results = []
    children = defaultdict(list)
    for msg in sv_messages.values():
        if msg["parent_id"] and msg["parent_id"] in sv_messages:
            children[msg["parent_id"]].append(msg["message_id"])

    # Find leaf nodes (no children or best-ranked children)
    leaves = [mid for mid in sv_messages if mid not in children]

    for leaf_id in leaves:
        chain = []
        current = leaf_id
        while current and current in sv_messages:
            chain.append(sv_messages[current])
            current = sv_messages[current]["parent_id"]
        chain.reverse()

        # Must start with user, alternate, and have at least one exchange
        if len(chain) < 2:
            continue
        if chain[0]["role"] != "user":
            continue

        messages = [{"role": m["role"], "content": m["text"]} for m in chain]

        # Validate alternation
        valid = True
        for i in range(1, len(messages)):
            if messages[i]["role"] == messages[i - 1]["role"]:
                valid = False
                break
        if not valid:
            continue

        results.append({
            "messages": messages,
            "source": source_name,
        })

    return results


def adapt_scandi_qa_sv(ds, source_name):
    """
    alexandrainst/scandi-qa sv split
    Columns: question, context, answers (struct with text list)
    """
    results = []
    for row in ds:
        question = normalize_text(row.get("question", ""))
        context = normalize_text(row.get("context", ""))
        answers = row.get("answers", {})

        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        # Skip if no valid answer
        if not answer_texts or not answer_texts[0]:
            continue

        answer = normalize_text(answer_texts[0])
        if not question or not answer:
            continue

        # Format as QA with context
        user_content = f"Givet följande text:\n{context}\n\nFråga: {question}" if context else question

        results.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ],
            "source": source_name,
        })
    return results


def adapt_swedish_instruct_gpt4(ds, source_name):
    """
    skvarre/swedish-instruct-data-chatgpt4
    Columns: human, gpt
    """
    results = []
    for row in ds:
        human = normalize_text(row.get("human", ""))
        gpt = normalize_text(row.get("gpt", ""))

        if not human or not gpt:
            continue
        if len(gpt) < 10:
            continue

        results.append({
            "messages": [
                {"role": "user", "content": human},
                {"role": "assistant", "content": gpt},
            ],
            "source": source_name,
        })
    return results


# Adapter registry
ADAPTERS = {
    "alpaca_swedish": adapt_alpaca_swedish,
    "wikipedia_qa_sv": adapt_wikipedia_qa_sv,
    "oasst2_sv": adapt_oasst2_sv,
    "scandi_qa_sv": adapt_scandi_qa_sv,
    "swedish_instruct_gpt4": adapt_swedish_instruct_gpt4,
}


def load_and_adapt(source_cfg, token=None):
    """Load a dataset and run its adapter."""
    from datasets import load_dataset

    name = source_cfg["name"]
    path = source_cfg["path"]
    subset = source_cfg.get("subset")
    split = source_cfg.get("split", "train")

    adapter = ADAPTERS.get(name)
    if adapter is None:
        print(f"  WARNING: No adapter for {name}, skipping")
        return []

    print(f"  Loading {name}: {path} subset={subset} split={split}")

    kwargs = {"split": split, "trust_remote_code": True}
    if subset:
        kwargs["name"] = subset
    if token:
        kwargs["token"] = token

    ds = load_dataset(path, **kwargs)
    results = adapter(ds, name)
    # Propagate license from config into each conversation record
    license_str = source_cfg.get("license", "unknown")
    for r in results:
        r["license"] = license_str
    print(f"  -> {len(results)} conversations from {name}")
    return results


# ─── Synthetic data generation ───────────────────────────────────────────────

SYNTHETIC_CATEGORIES = [
    ("casual_chat", "Vardaglig konversation: hälsning, small talk, planering av vardagliga aktiviteter"),
    ("practical_advice", "Praktiska råd: matlagning, hushåll, trädgård, bilreparation, hemfix"),
    ("customer_support", "Kundtjänst: klagomål, returer, fakturor, leveransproblem, teknisk support"),
    ("small_business", "Småföretag: bokföring, momsdeklaration, anställning, marknadsföring"),
    ("email_drafting", "E-postutkast: formella brev, ansökningar, tackbrev, förfrågningar"),
    ("sms_replies", "SMS/chatt-svar: korta snabba svar, talspråk, förkortningar"),
    ("planning", "Planering: resor, semester, evenemang, möten, projektplanering"),
    ("parenting_school", "Föräldraskap/skola: barnuppfostran, läxhjälp, skolval, fritidsaktiviteter"),
    ("polite_refusals", "Artiga avböjanden: tacka nej till inbjudningar, förfrågningar, erbjudanden"),
    ("clarifications", "Förtydliganden: be om mer information, klargöra missförstånd"),
    ("multi_turn", "Flerstegssamtal: uppföljningsfrågor, fördjupning, ämnesbyte"),
    ("factual_explanation", "Faktaförklaring: vetenskap, historia, geografi, samhälle, svensk kultur"),
]

SYNTHETIC_STYLES = [
    "formell (korrekt, högtidlig, skriftspråklig)",
    "neutral (vardaglig men korrekt)",
    "vardaglig (talspråklig, avslappnad)",
    "koncis (kort och kärnfull)",
    "förklarande (utförlig, pedagogisk)",
    "varm (empatisk, inkännande)",
    "saklig (rak, professionell)",
    "praktisk (handlingsorienterad, steg-för-steg)",
]

SYNTHETIC_SYSTEM_PROMPT = """\
Du är en datagenereringsassistent som skapar naturliga svenska konversationer \
för att träna en språkmodell.

REGLER:
- Skriv idiomatisk svenska, INTE översatt engelska
- Använd naturliga svenska uttryck, ordföljd och partikelverb
- Variera meningslängd och komplexitet
- Inkludera svenska kulturella referenser där det passar \
(personnummer, BankID, fika, allemansrätten, etc.)
- Undvik amerikansk/engelsk kulturell kontext
- Undvik policyformuleringar och disclaimers
- Assistenten ska vara hjälpsam, naturlig och lagom formell/informell beroende på stilen
- Varje konversation ska ha minst 2 och max 8 meddelanden (user/assistant alternerande)

Returnera EXAKT en JSON-array med konversationer i detta format:
[
  {
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
]

Returnera ENBART JSON, inget annat.\
"""


def generate_synthetic_conversations(target, api_key, seed=42):
    """Generate synthetic Swedish conversations using the Anthropic API.

    Requires: pip install anthropic

    Args:
        target: Number of conversations to generate.
        api_key: Anthropic API key (ANTHROPIC_API_KEY).
        seed: Random seed for category/style selection.

    Returns:
        List of {"messages": [...], "source": "synthetic", "license": "synthetic"}.
    """
    try:
        import anthropic
    except ImportError:
        print("ERROR: 'anthropic' package required for synthetic generation.")
        print("  Install with: pip install anthropic")
        return []

    client = anthropic.Anthropic(api_key=api_key)
    rng = random.Random(seed)

    results = []
    batch_size = 5
    num_batches = (target + batch_size - 1) // batch_size

    print(f"\n  Generating {target} synthetic conversations ({num_batches} API calls)...")

    for batch_idx in range(num_batches):
        if len(results) >= target:
            break

        cat_id, cat_desc = rng.choice(SYNTHETIC_CATEGORIES)
        style = rng.choice(SYNTHETIC_STYLES)
        remaining = min(batch_size, target - len(results))

        user_prompt = (
            f"Generera {remaining} svenska konversationer.\n"
            f"Kategori: {cat_desc}\n"
            f"Stil: {style}\n"
            f"Varje konversation ska vara unik och naturlig."
        )

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                system=SYNTHETIC_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            text = response.content[0].text.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]

            convs = json.loads(text)
            if not isinstance(convs, list):
                convs = [convs]

            for conv in convs:
                messages = conv.get("messages", [])
                if not messages:
                    continue
                results.append({
                    "messages": messages,
                    "source": "synthetic",
                    "license": "synthetic",
                })

            if (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
                print(f"  Progress: {len(results)}/{target} ({batch_idx + 1}/{num_batches} batches)")

        except json.JSONDecodeError as e:
            print(f"  WARNING: JSON parse error in batch {batch_idx}: {e}")
            continue
        except Exception as e:
            print(f"  WARNING: API error in batch {batch_idx}: {e}")
            continue

    print(f"  Generated {len(results)} synthetic conversations")
    return results[:target]


def validate_conversation(conv):
    """Validate a single conversation dict."""
    messages = conv.get("messages", [])
    if len(messages) < 2:
        return False
    if messages[0]["role"] != "user":
        return False
    if messages[-1]["role"] != "assistant":
        return False
    for msg in messages:
        if not msg.get("content", "").strip():
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data for Swedish LFM2.5")
    parser.add_argument("--config", required=True, help="Path to datasets.yaml")
    parser.add_argument("--out", required=True, help="Output directory for manifests")
    parser.add_argument("--eval_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_only", action="store_true")
    parser.add_argument("--synthetic_target", type=int, default=0,
                        help="Number of synthetic conversations to generate via Anthropic API")
    args = parser.parse_args()

    random.seed(args.seed)
    config = load_config(args.config)
    eval_ratio = args.eval_ratio or config.get("splits", {}).get("sft_eval_ratio", 0.05)

    token = os.environ.get("HF_TOKEN")

    # Collect all conversations
    all_convs = []
    for src in config.get("sft_sources", []):
        if args.local_only:
            print(f"  Skipping HF source: {src['name']} (--local_only)")
            continue
        try:
            convs = load_and_adapt(src, token=token)
            all_convs.extend(convs)
        except Exception as e:
            print(f"  ERROR loading {src['name']}: {e}")
            continue

    # Generate synthetic conversations if requested
    if args.synthetic_target > 0:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_key:
            print("ERROR: --synthetic_target requires ANTHROPIC_API_KEY environment variable")
            return 1
        print(f"\nGenerating synthetic data (target={args.synthetic_target})...")
        synthetic = generate_synthetic_conversations(
            target=args.synthetic_target,
            api_key=anthropic_key,
            seed=args.seed,
        )
        all_convs.extend(synthetic)

    print(f"\nTotal raw conversations: {len(all_convs)}")

    # Validate
    before = len(all_convs)
    all_convs = [c for c in all_convs if validate_conversation(c)]
    print(f"After validation: {before} -> {len(all_convs)}")

    # Deduplicate
    seen = set()
    deduped = []
    for conv in all_convs:
        h = messages_hash(conv["messages"])
        if h not in seen:
            seen.add(h)
            deduped.append(conv)
    print(f"After dedup: {len(all_convs)} -> {len(deduped)} ({len(all_convs) - len(deduped)} duplicates)")
    all_convs = deduped

    # Split: stratified by source — sample eval proportionally from each source
    by_source = defaultdict(list)
    for conv in all_convs:
        by_source[conv["source"]].append(conv)
    train_convs = []
    eval_convs = []
    for source, convs in by_source.items():
        random.shuffle(convs)
        n_eval = max(1, int(len(convs) * eval_ratio))
        eval_convs.extend(convs[:n_eval])
        train_convs.extend(convs[n_eval:])
    random.shuffle(train_convs)
    random.shuffle(eval_convs)

    # Write manifests
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, items):
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_path = out_dir / "sft_train.jsonl"
    eval_path = out_dir / "sft_eval.jsonl"
    write_jsonl(train_path, train_convs)
    write_jsonl(eval_path, eval_convs)

    # Stats
    source_counts = Counter(c["source"] for c in all_convs)
    total_turns = sum(len(c["messages"]) for c in all_convs)
    print(f"\n=== SFT Data Stats ===")
    print(f"Total conversations: {len(all_convs):,}")
    print(f"Train: {len(train_convs):,}")
    print(f"Eval: {len(eval_convs):,}")
    print(f"Total turns: {total_turns:,}")
    print(f"\nSource distribution:")
    for src, count in source_counts.most_common():
        print(f"  {src}: {count:,}")
    print(f"\nManifests written to: {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
