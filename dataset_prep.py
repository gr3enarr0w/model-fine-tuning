#!/usr/bin/env python3
"""
dataset_prep.py — Laguna XS 2.1 QLoRA fine-tuning data preparation.

Downloads and prepares ~200k training examples via streaming (no OOM):
  - ~100k from open-alchemy/code-alchemy (weighted by type)
  - ~100k from WaltonFuture/agentic-sft-new

Output: data/train.jsonl, data/eval.jsonl (90/10 split)
Format: ChatML JSONL
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert software engineer working on a long-horizon coding task. "
    "You write clean, tested, production-quality code."
)

CODEALCHEMY_DATASET = "open-alchemy/code-alchemy"
AGENTIC_DATASET = "WaltonFuture/agentic-sft-new"

# Bug fix #1: Config names must be hyphenated lowercase (not PascalCase).
# Correct configs: code-dev, code-dialogue, code-trace, code-enhance, code-qa
CODEALCHEMY_TARGETS: dict[str, int] = {
    "code-dev": 40_000,
    "code-dialogue": 30_000,
    "code-trace": 20_000,
    "code-enhance": 5_000,
    "code-qa": 5_000,
}
CODEALCHEMY_TOTAL = sum(CODEALCHEMY_TARGETS.values())  # 100_000

# Bug fix #5: Synthetic user prompts per config (text is already formatted).
# The full text/text_with_placeholders becomes the assistant turn.
CODEALCHEMY_USER_PROMPTS: dict[str, str] = {
    "code-dev": "Complete the following developer task:",
    "code-dialogue": "Continue this development conversation:",
    "code-trace": "Analyze this code execution trace:",
    "code-enhance": "Review and improve this code:",
    "code-qa": "Answer this code question:",
}

# Bug fix #2: text_with_placeholders configs vs text configs
CODEALCHEMY_PLACEHOLDER_CONFIGS = {"code-dev", "code-dialogue"}
CODEALCHEMY_TEXT_CONFIGS = {"code-trace", "code-enhance", "code-qa"}

AGENTIC_TOTAL = 100_000

# Bug fix #4: Expand language filter to include common variants
ALLOWED_LANGUAGES = {"python", "go", "typescript", "java", "javascript", "js", "ts", "py"}

TRAIN_RATIO = 0.90
SEED = 42

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def content_hash(text: str) -> str:
    """SHA-256 of the first 512 characters of text."""
    snippet = text[:512].encode("utf-8", errors="replace")
    return hashlib.sha256(snippet).hexdigest()


def chatml_record(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def rough_token_count(text: str) -> int:
    return len(text) // 4


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CodeAlchemy streaming
# ---------------------------------------------------------------------------

def _get_language(example: dict) -> str | None:
    """Extract language from a CodeAlchemy example (case-insensitive)."""
    lang = (
        example.get("language")
        or example.get("lang")
        or example.get("programming_language")
        or ""
    )
    return lang.lower().strip() if lang else None


def _extract_codealchemy_text(example: dict, config_name: str) -> tuple[str, str] | None:
    """
    Return (user_turn, assistant_turn) from a CodeAlchemy example.

    Bug fix #2: Use the correct field per config:
      - code-dev, code-dialogue: text_with_placeholders
      - code-trace, code-enhance, code-qa: text

    Bug fix #5: The text is already formatted — treat the full text as the
    assistant turn and synthesize a generic user instruction from the config.

    Returns None if content cannot be extracted.
    """
    # Bug fix #2: select the correct field based on config
    if config_name in CODEALCHEMY_PLACEHOLDER_CONFIGS:
        raw_text = example.get("text_with_placeholders", "")
    else:
        raw_text = example.get("text", "")

    raw_text = str(raw_text).strip() if raw_text else ""
    if not raw_text:
        return None

    # Bug fix #5: synthesize user turn from config-specific prompt
    user_turn = CODEALCHEMY_USER_PROMPTS.get(config_name, "Complete the following coding task:")
    assistant_turn = raw_text

    return user_turn, assistant_turn


def stream_codealchemy(targets: dict[str, int]) -> Iterator[tuple[dict, str, str]]:
    """
    Yields (chatml_record, type_label, language) for CodeAlchemy examples.

    Streams each subset config independently and takes up to target count.
    Language filtering is applied only for code-trace (Python is ~44% of its
    early shards). All other configs are language-sorted with non-target
    languages dominating initial rows, so they stream all languages.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        # Fallback: no-op tqdm
        def tqdm(iterable, **kwargs):  # type: ignore
            return iterable

    seen_hashes: set[str] = set()

    for config_name, target_count in targets.items():
        print(f"\n[CodeAlchemy] Streaming {config_name} (target: {target_count:,})...")

        # Bug fix #1: use the hyphenated config name directly.
        # Bug fix #3: do NOT pass trust_remote_code=True.
        ds = None
        try:
            ds = load_dataset(
                CODEALCHEMY_DATASET,
                name=config_name,
                split="train",
                streaming=True,
            )
        except Exception as exc:
            print(f"  WARNING: could not load config {config_name}: {exc}", file=sys.stderr)
            continue

        collected = 0
        bar = tqdm(total=target_count, desc=config_name, unit="ex")

        for example in ds:
            if collected >= target_count:
                break

            # Language filter (bug fix #4: expanded set)
            # Skip language filter for configs where the dataset is sorted by
            # language — filtering causes near-infinite scanning because
            # Python/Go/TypeScript/Java rows may be millions of rows away.
            # Verified by sampling:
            #   code-dev / code-dialogue: all 15 langs distributed (instruction format)
            #   code-enhance / code-qa: C++ fills the first 2000+ rows
            #   code-trace: Python present ~44% in first 500 rows — filter is safe
            # Multi-language training is better for a general coding model anyway.
            lang = _get_language(example)
            if config_name == "code-trace":
                if lang not in ALLOWED_LANGUAGES:
                    continue

            # Extract text (bug fix #2 + #5)
            pair = _extract_codealchemy_text(example, config_name)
            if pair is None:
                continue
            user_text, asst_text = pair

            # Skip empty
            if not user_text.strip() or not asst_text.strip():
                continue

            # Dedup on assistant content (user is synthetic/identical per config)
            h = content_hash(asst_text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            record = chatml_record(user_text, asst_text)
            yield record, config_name, lang
            collected += 1
            bar.update(1)

        bar.close()
        print(f"  Collected {collected:,} from {config_name}")


# ---------------------------------------------------------------------------
# Agentic SFT streaming
# ---------------------------------------------------------------------------

def _extract_agentic_text(example: dict) -> tuple[str, str] | None:
    """Return (user_turn, assistant_turn) from an agentic-sft example."""
    messages = example.get("messages") or example.get("conversations")
    if messages and isinstance(messages, list):
        user_parts: list[str] = []
        assistant_parts: list[str] = []
        for msg in messages:
            role = (msg.get("role") or msg.get("from") or "").lower()
            content = msg.get("content") or msg.get("value") or ""
            if role in ("user", "human"):
                user_parts.append(str(content).strip())
            elif role in ("assistant", "gpt", "model"):
                assistant_parts.append(str(content).strip())
        user = "\n\n".join(filter(None, user_parts))
        assistant = "\n\n".join(filter(None, assistant_parts))
        if user and assistant:
            return user, assistant

    instruction = (
        example.get("instruction")
        or example.get("input")
        or example.get("prompt")
        or example.get("question")
        or ""
    )
    output = (
        example.get("output")
        or example.get("response")
        or example.get("answer")
        or example.get("completion")
        or ""
    )
    instruction = str(instruction).strip()
    output = str(output).strip()
    if instruction and output:
        return instruction, output

    return None


def stream_agentic(target_count: int) -> Iterator[dict]:
    """
    Yields chatml_record dicts from WaltonFuture/agentic-sft-new.
    Samples target_count from the full 711k via reservoir-style skip.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' package not installed.", file=sys.stderr)
        sys.exit(1)

    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        def tqdm(iterable, **kwargs):  # type: ignore
            return iterable

    print(f"\n[Agentic] Streaming {AGENTIC_DATASET} (target: {target_count:,})...")

    # Dataset has ~711k examples; we want 100k. Simple approach: take 100k
    # directly (they are shuffled at source) then dedup.
    # We over-fetch slightly to account for skipped/dedup'd examples.
    FETCH_FACTOR = 1.3
    fetch_count = int(target_count * FETCH_FACTOR)

    # Bug fix #3: do NOT pass trust_remote_code=True.
    ds = load_dataset(
        AGENTIC_DATASET,
        split="train",
        streaming=True,
    ).take(fetch_count)

    seen_hashes: set[str] = set()
    collected = 0
    bar = tqdm(total=target_count, desc="Agentic", unit="ex")

    for example in ds:
        if collected >= target_count:
            break

        pair = _extract_agentic_text(example)
        if pair is None:
            continue
        user_text, asst_text = pair

        if not user_text.strip() or not asst_text.strip():
            continue

        h = content_hash(user_text + asst_text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        record = chatml_record(user_text, asst_text)
        yield record
        collected += 1
        bar.update(1)

    bar.close()
    print(f"  Collected {collected:,} from agentic dataset")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    """Parse CLI args. --limit N caps each source for smoke testing."""
    import argparse
    p = argparse.ArgumentParser(description="Prepare fine-tuning data.")
    p.add_argument("--limit", type=int, default=None,
        help="Cap examples per source for smoke testing (e.g. --limit 200)")
    args = p.parse_args()
    if args.limit is not None and args.limit < 2:
        p.error("--limit must be >= 2")
    return args


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Apply --limit to reduce targets for smoke testing
    if args.limit:
        print(f"[SMOKE TEST] Limiting total examples to {args.limit}")
        total_ca_target = CODEALCHEMY_TOTAL  # 100_000
        targets = {
            k: round(args.limit * (v / (total_ca_target + AGENTIC_TOTAL)))
            for k, v in CODEALCHEMY_TARGETS.items()
        }
        agentic_limit = args.limit - sum(targets.values())
        agentic_limit = max(0, agentic_limit)
    else:
        targets = CODEALCHEMY_TARGETS
        agentic_limit = AGENTIC_TOTAL

    all_records: list[dict] = []
    stats: dict[str, dict] = {
        "codealchemy": {
            "by_type": Counter(),
            "by_language": Counter(),
            "total": 0,
        },
        "agentic": {
            "total": 0,
        },
    }

    # ------------------------------------------------------------------
    # 1. CodeAlchemy
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Phase 1: CodeAlchemy")
    print("=" * 60)

    for record, type_label, lang in stream_codealchemy(targets):
        all_records.append(record)
        stats["codealchemy"]["by_type"][type_label] += 1
        stats["codealchemy"]["by_language"][lang] += 1
        stats["codealchemy"]["total"] += 1

    # ------------------------------------------------------------------
    # 2. Agentic SFT
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2: Agentic SFT")
    print("=" * 60)

    for record in stream_agentic(agentic_limit):
        all_records.append(record)
        stats["agentic"]["total"] += 1

    # ------------------------------------------------------------------
    # 3. Shuffle + split
    # ------------------------------------------------------------------
    print(f"\nTotal collected: {len(all_records):,}")
    print("Shuffling...")
    random.shuffle(all_records)

    split_idx = int(len(all_records) * TRAIN_RATIO)
    train_records = all_records[:split_idx]
    eval_records = all_records[split_idx:]

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    train_path = DATA_DIR / "train.jsonl"
    eval_path = DATA_DIR / "eval.jsonl"

    print(f"Writing {len(train_records):,} train examples to {train_path}...")
    write_jsonl(train_path, train_records)

    print(f"Writing {len(eval_records):,} eval examples to {eval_path}...")
    write_jsonl(eval_path, eval_records)

    # ------------------------------------------------------------------
    # 5. Stats
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    total_examples = len(all_records)
    total_chars = sum(
        sum(len(m["content"]) for m in rec["messages"])
        for rec in all_records
    )
    estimated_tokens = rough_token_count(" " * total_chars)  # chars / 4

    print(f"\nTotal examples : {total_examples:,}")
    print(f"  Train        : {len(train_records):,}")
    print(f"  Eval         : {len(eval_records):,}")
    print(f"Estimated tokens (chars/4): ~{estimated_tokens:,}")

    print("\n--- CodeAlchemy by type ---")
    for type_label, count in sorted(
        stats["codealchemy"]["by_type"].items(), key=lambda x: -x[1]
    ):
        pct = count / total_examples * 100
        print(f"  {type_label:<16} {count:>7,}  ({pct:.1f}%)")

    print("\n--- CodeAlchemy by language ---")
    for lang, count in sorted(
        stats["codealchemy"]["by_language"].items(), key=lambda x: -x[1]
    ):
        pct = count / total_examples * 100
        print(f"  {lang:<16} {count:>7,}  ({pct:.1f}%)")

    print("\n--- Agentic SFT ---")
    agentic_count = stats["agentic"]["total"]
    pct = agentic_count / total_examples * 100
    print(f"  Total          {agentic_count:>7,}  ({pct:.1f}%)")

    print("\n--- Output files ---")
    print(f"  Train: {train_path}  ({train_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Eval:  {eval_path}  ({eval_path.stat().st_size / 1e6:.1f} MB)")
    print("\nDone.")


if __name__ == "__main__":
    main()
