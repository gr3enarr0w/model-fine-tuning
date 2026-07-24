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

# Weighted targets from CodeAlchemy
CODEALCHEMY_TARGETS: dict[str, int] = {
    "CodeDev": 40_000,
    "CodeDialogue": 30_000,
    "CodeTrace": 20_000,
    "CodeEnhance": 5_000,
    "CodeQA": 5_000,
}
CODEALCHEMY_TOTAL = sum(CODEALCHEMY_TARGETS.values())  # 100_000

AGENTIC_TOTAL = 100_000

ALLOWED_LANGUAGES = {"python", "go", "typescript", "java"}

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


def _extract_codealchemy_text(example: dict) -> tuple[str, str] | None:
    """
    Return (user_turn, assistant_turn) from a CodeAlchemy example.
    Returns None if content cannot be extracted.
    """
    # Try conversation-style fields first
    messages = example.get("messages") or example.get("conversations")
    if messages and isinstance(messages, list):
        # Filter to user/assistant pairs
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

    # Flat instruction/output style
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


def stream_codealchemy(targets: dict[str, int]) -> Iterator[tuple[dict, str, str]]:
    """
    Yields (chatml_record, type_label, language) for CodeAlchemy examples.

    Streams each subset type independently and takes up to target count,
    filtering to allowed languages.
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

    for type_label, target_count in targets.items():
        print(f"\n[CodeAlchemy] Streaming {type_label} (target: {target_count:,})...")

        # Try loading with a config/split that matches the type label.
        # The dataset may expose types as configs, splits, or a 'type' column.
        # We try multiple strategies gracefully.
        ds = None

        # Strategy 1: type_label as config name
        try:
            ds = load_dataset(
                CODEALCHEMY_DATASET,
                name=type_label,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception:
            pass

        # Strategy 2: flat dataset, filter by 'type' column client-side
        if ds is None:
            try:
                ds = load_dataset(
                    CODEALCHEMY_DATASET,
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )
                ds = ds.filter(
                    lambda ex: (ex.get("type") or ex.get("task_type") or "").strip() == type_label
                )
            except Exception as exc:
                print(f"  WARNING: could not load {type_label}: {exc}", file=sys.stderr)
                continue

        collected = 0
        bar = tqdm(total=target_count, desc=type_label, unit="ex")

        for example in ds:
            if collected >= target_count:
                break

            # Language filter
            lang = _get_language(example)
            if lang not in ALLOWED_LANGUAGES:
                continue

            # Extract text
            pair = _extract_codealchemy_text(example)
            if pair is None:
                continue
            user_text, asst_text = pair

            # Skip empty
            if not user_text.strip() or not asst_text.strip():
                continue

            # Dedup
            h = content_hash(user_text + asst_text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            record = chatml_record(user_text, asst_text)
            yield record, type_label, lang
            collected += 1
            bar.update(1)

        bar.close()
        print(f"  Collected {collected:,} from {type_label}")


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

    ds = load_dataset(
        AGENTIC_DATASET,
        split="train",
        streaming=True,
        trust_remote_code=True,
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

def main() -> None:
    random.seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

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

    for record, type_label, lang in stream_codealchemy(CODEALCHEMY_TARGETS):
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

    for record in stream_agentic(AGENTIC_TOTAL):
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
