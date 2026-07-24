#!/usr/bin/env python3
"""
benchmark.py — Strategy A (per-language) vs Strategy B (generalist) evaluation.

Evaluates on data/test.jsonl (held out — never seen during training).
Queries Modal endpoints for fine-tuned models, Ollama for baseline.

Usage:
    python benchmark.py [--test-set data/test.jsonl] [--username <modal-user>]
                        [--output-dir results] [--max-per-lang N]
                        [--strategies a b baseline] [--languages python go ...]
"""

import argparse
import json
import re
import statistics
import time
import urllib.request
import urllib.error
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGUAGES: list[str] = [
    "python", "javascript", "go", "rust", "sql", "markdown",
    "typescript", "shell", "java", "php", "c", "cpp", "csharp", "ruby", "swift",
]

OLLAMA_BASE = "http://localhost:11434"

# Modal FastAPI app name template — fill with your Modal username.
# Strategy A: per-language adapters served as separate models.
# Strategy B: single generalist adapter.
MODAL_BASE = "https://{username}--gemma-e4b-codealchemy-fastapi-app.modal.run"

# Ollama model name for the untuned baseline.
BASELINE_OLLAMA_MODEL = "gemma4:4b-it-qat"

# Heuristic keyword lists for language detection when no metadata field exists.
# Ordered from most-specific to avoid cross-matching (e.g. "typescript" before "javascript").
_LANG_KEYWORDS: dict[str, list[str]] = {
    "typescript": ["typescript", ".ts", "tsx", "interface ", "type ", ": string", ": number", ": boolean"],
    "javascript": ["javascript", ".js", "jsx", "require(", "module.exports", "const ", "let ", "var "],
    "python":     ["python", ".py", "def ", "import ", "print(", "if __name__"],
    "go":         ["golang", " go ", ".go", "func ", "package main", ":= ", "fmt."],
    "rust":       ["rust", ".rs", "fn main", "let mut", "impl ", "pub fn", "use std"],
    "sql":        ["sql", "select ", "insert into", "create table", "from ", "where ", "join "],
    "markdown":   ["markdown", ".md", "## ", "### ", "```", "**", "__"],
    "shell":      ["bash", "shell", "#!/bin/sh", "#!/bin/bash", "echo ", "export ", "grep "],
    "java":       ["java", ".java", "public class", "void main", "System.out", "@Override"],
    "php":        ["php", ".php", "<?php", "echo ", "$_GET", "$_POST", "->"],
    "c":          [" c ", ".c ", "#include <", "int main(", "printf(", "malloc(", "sizeof("],
    "cpp":        ["c++", ".cpp", "#include <", "std::", "cout ", "cin ", "namespace"],
    "csharp":     ["c#", ".cs", "using System", "namespace ", "static void Main", "Console."],
    "ruby":       ["ruby", ".rb", "def ", "end\n", "puts ", "require '", "attr_"],
    "swift":      ["swift", ".swift", "func ", "var ", "let ", "print(", "import Foundation"],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark configuration."""
    parser = argparse.ArgumentParser(
        description="Benchmark Strategy A (per-language Gemma E4B) vs Strategy B (generalist) vs baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test-set",
        default="data/test.jsonl",
        help="Path to held-out test JSONL file (default: data/test.jsonl).",
    )
    parser.add_argument(
        "--username",
        default="",
        help="Modal username for constructing endpoint URLs (e.g. 'myuser').",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to write benchmark_{timestamp}.json/.md (default: results/).",
    )
    parser.add_argument(
        "--max-per-lang",
        type=int,
        default=0,
        help="Max examples to evaluate per language (0 = all, default: 0).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["a", "b", "baseline"],
        default=["a", "b", "baseline"],
        help="Which strategies to run (default: a b baseline).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=LANGUAGES,
        default=LANGUAGES,
        help="Languages to evaluate (default: all 15).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds per request (default: 60).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual model calls; emit synthetic metrics for testing output format.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_set(path: str = "data/test.jsonl") -> list[dict]:
    """Load held-out test examples from JSONL.

    Each line must be a JSON object with at minimum a ``messages`` key
    containing a list of ChatML-format message dicts.  A ``language``
    field is used when present; otherwise language is inferred from the
    message content via :func:`infer_language`.

    Args:
        path: Filesystem path to the JSONL file.

    Returns:
        List of parsed record dicts, one per non-empty line.

    Raises:
        FileNotFoundError: If the file does not exist at *path*.
        json.JSONDecodeError: If any line is not valid JSON.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Test set not found: {p.resolve()}")
    records: list[dict] = []
    with p.open() as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {lineno}: {exc.msg}", exc.doc, exc.pos
                ) from exc
    return records


def infer_language(record: dict) -> str:
    """Infer the programming language of a test record from its content.

    Checks the ``language`` field first, then falls back to keyword
    heuristics applied to the concatenated text of all message turns.

    Args:
        record: A single ChatML record dict with a ``messages`` list.

    Returns:
        Detected language string (lowercase, matching one of
        :data:`LANGUAGES`), or ``"unknown"`` if no match is found.
    """
    # Prefer explicit metadata field.
    for field in ("language", "lang", "programming_language"):
        val = record.get(field, "")
        if val:
            normalized = val.lower().strip()
            if normalized in LANGUAGES:
                return normalized

    # Fall back to content scanning.
    content = " ".join(
        msg.get("content", "") for msg in record.get("messages", [])
    ).lower()

    for lang in LANGUAGES:
        keywords = _LANG_KEYWORDS.get(lang, [lang])
        if any(kw.lower() in content for kw in keywords):
            return lang

    return "unknown"


def filter_by_language(records: list[dict], language: str) -> list[dict]:
    """Filter test records to those matching a specific language.

    Language is determined by :func:`infer_language` for each record.

    Args:
        records: Full list of test records.
        language: Target language string (e.g. ``"python"``).

    Returns:
        Subset of *records* whose inferred language matches *language*.
    """
    return [r for r in records if infer_language(r) == language]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_post(url: str, payload: dict, timeout: float = 60.0) -> dict:
    """POST JSON to *url* and return the parsed response body.

    Args:
        url: Full HTTP/HTTPS endpoint URL.
        payload: Dict to serialize as the request body.
        timeout: Socket timeout in seconds.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        urllib.error.URLError: On network or HTTP errors.
    """
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def query_ollama(model: str, messages: list[dict], timeout: float = 60.0) -> tuple[str, float]:
    """Query a local Ollama model via its OpenAI-compatible endpoint.

    Args:
        model: Ollama model name (e.g. ``"gemma4:4b-it-qat"``).
        messages: ChatML-format message list.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of ``(response_text, latency_ms)`` where *latency_ms* is
        the wall-clock time for the full round-trip in milliseconds.
    """
    url = f"{OLLAMA_BASE}/v1/chat/completions"
    payload = {"model": model, "messages": messages, "stream": False}
    t0 = time.perf_counter()
    body = _http_post(url, payload, timeout=timeout)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    text = body["choices"][0]["message"]["content"]
    return text, latency_ms


def query_model(
    endpoint: str,
    messages: list[dict],
    api_key: str = "sk-local",
    timeout: float = 60.0,
) -> tuple[str, float]:
    """Query an OpenAI-compatible endpoint and return (response_text, latency_ms).

    Sends a non-streaming ``/v1/chat/completions`` request.  The *endpoint*
    should be the base URL (without the path suffix); the ``/v1/chat/completions``
    suffix is appended automatically.

    Args:
        endpoint: Base URL of the OpenAI-compatible server, e.g.
            ``"https://user--app.modal.run"`` or ``"http://localhost:11434"``.
        messages: ChatML-format message list.
        api_key: Bearer token sent in the ``Authorization`` header
            (default ``"sk-local"`` for self-hosted endpoints).
        timeout: Request timeout in seconds.

    Returns:
        Tuple of ``(response_text, latency_ms)``.

    Raises:
        urllib.error.URLError: On network or HTTP errors.
    """
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": messages,
        "stream": False,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    latency_ms = (time.perf_counter() - t0) * 1000.0
    text = body["choices"][0]["message"]["content"]
    return text, latency_ms


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Tokenize *text* into a bag of lowercase word tokens.

    Splits on whitespace and strips punctuation from token boundaries.
    Used by :func:`compute_loss` for token-overlap F1.

    Args:
        text: Raw text string.

    Returns:
        Set of cleaned token strings.
    """
    return set(re.findall(r"\b\w+\b", text.lower()))


def compute_loss(model_response: str, expected: str) -> float:
    """Compute a simple token-overlap proxy loss (1 − F1).

    Real validation perplexity requires per-token log-probabilities
    (``logprobs``) from the model, which are not available from all
    endpoints without extra configuration.  This function uses
    token-overlap F1 between the model output and the expected
    (gold) assistant turn as a fast, dependency-free proxy.

    Loss ranges from 0.0 (perfect overlap) to 1.0 (no overlap).

    Args:
        model_response: Text produced by the model.
        expected: Gold / reference assistant text.

    Returns:
        Float loss in ``[0.0, 1.0]``.  Returns ``1.0`` when *expected*
        is empty to avoid division by zero.
    """
    if not expected:
        return 1.0
    pred_tokens = _tokenize(model_response)
    gold_tokens = _tokenize(expected)
    if not pred_tokens or not gold_tokens:
        return 1.0
    tp = len(pred_tokens & gold_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(gold_tokens)
    if precision + recall == 0:
        return 1.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(1.0 - f1, 4)


def latency_per_token_ms(latency_ms: float, response_text: str) -> float:
    """Estimate per-token latency in milliseconds.

    Divides total round-trip latency by an approximate token count
    (characters / 4, clamped to ≥ 1) because most endpoints do not
    return token counts in non-streaming mode.

    Args:
        latency_ms: Total request round-trip time in milliseconds.
        response_text: The generated text (used to estimate token count).

    Returns:
        Estimated milliseconds per token (float).
    """
    approx_tokens = max(1, len(response_text) // 4)
    return round(latency_ms / approx_tokens, 3)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _extract_gold(record: dict) -> tuple[list[dict], str]:
    """Split a ChatML record into prompt messages and gold assistant text.

    Strips the last ``assistant`` turn from the message list to form
    the input prompt; the content of that turn becomes the gold reference.

    Args:
        record: A single ChatML record with a ``messages`` list.

    Returns:
        Tuple of ``(prompt_messages, gold_text)`` where *gold_text* is
        an empty string when no assistant turn is present.
    """
    messages: list[dict] = record.get("messages", [])
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1], messages[-1].get("content", "")
    return messages, ""


def _make_endpoint(strategy: str, language: str, username: str) -> str:
    """Construct the inference endpoint URL for a given strategy.

    Strategy A uses per-language Modal endpoints named
    ``gemma-e4b-{language}-fastapi-app``.
    Strategy B uses a single generalist Modal endpoint named
    ``gemma-e4b-codealchemy-fastapi-app``.
    Baseline queries the local Ollama server.

    Args:
        strategy: One of ``"a"``, ``"b"``, or ``"baseline"``.
        language: Target language (used only for strategy A).
        username: Modal username to interpolate into the URL template.

    Returns:
        Full base URL string for the endpoint.
    """
    if strategy == "baseline":
        return OLLAMA_BASE
    if strategy == "b":
        return MODAL_BASE.format(username=username)
    # Strategy A: per-language endpoint
    return f"https://{username}--gemma-e4b-{language}-fastapi-app.modal.run"


def run_benchmark(
    strategy: str,
    language: str,
    records: list[dict],
    username: str = "",
    max_per_lang: int = 0,
    timeout: float = 60.0,
    dry_run: bool = False,
) -> dict:
    """Run evaluation for one strategy on one language; return metrics dict.

    Iterates over language-filtered *records*, queries the appropriate
    model endpoint, and aggregates per-example losses and latencies.

    Args:
        strategy: ``"a"`` (per-language Modal), ``"b"`` (generalist Modal),
            or ``"baseline"`` (untuned Ollama).
        language: Target language string.
        records: Full test set (will be filtered internally).
        username: Modal username (required for strategies A and B).
        max_per_lang: Maximum examples to evaluate per language (0 = all).
        timeout: Per-request HTTP timeout in seconds.
        dry_run: When ``True``, skip model calls and return synthetic metrics.

    Returns:
        Dict with keys ``strategy``, ``language``, ``n``, ``mean_loss``,
        ``median_latency_per_token_ms``, ``p95_latency_per_token_ms``,
        ``errors``, and ``examples`` (list of per-example dicts).
    """
    subset = filter_by_language(records, language)
    if max_per_lang and max_per_lang > 0:
        subset = subset[:max_per_lang]

    endpoint = _make_endpoint(strategy, language, username)
    losses: list[float] = []
    latencies_pt: list[float] = []
    errors: list[str] = []
    examples: list[dict] = []

    for i, record in enumerate(subset):
        prompt_msgs, gold_text = _extract_gold(record)

        if dry_run:
            # Synthetic values for format validation.
            loss = round(0.3 + i * 0.01, 4)
            latency_ms = 200.0 + i * 10.0
            response_text = f"[dry-run] {language} response {i}"
        else:
            try:
                if strategy == "baseline":
                    response_text, latency_ms = query_ollama(
                        BASELINE_OLLAMA_MODEL, prompt_msgs, timeout=timeout
                    )
                else:
                    response_text, latency_ms = query_model(
                        endpoint, prompt_msgs, timeout=timeout
                    )
            except Exception as exc:  # noqa: BLE001
                err_msg = f"example {i}: {type(exc).__name__}: {exc}"
                errors.append(err_msg)
                examples.append({"i": i, "error": err_msg})
                continue
            loss = compute_loss(response_text, gold_text)

        lpt = latency_per_token_ms(latency_ms, response_text)
        losses.append(loss)
        latencies_pt.append(lpt)
        examples.append(
            {
                "i": i,
                "loss": loss,
                "latency_ms": round(latency_ms, 1),
                "latency_per_token_ms": lpt,
            }
        )

    n = len(losses)
    return {
        "strategy": strategy,
        "language": language,
        "endpoint": endpoint,
        "n": n,
        "mean_loss": round(statistics.mean(losses), 4) if n else None,
        "median_latency_per_token_ms": round(statistics.median(latencies_pt), 3) if n else None,
        "p95_latency_per_token_ms": (
            round(sorted(latencies_pt)[int(len(latencies_pt) * 0.95)], 3)
            if n >= 2 else (round(latencies_pt[0], 3) if n == 1 else None)
        ),
        "errors": errors,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _markdown_table(results: list[dict]) -> str:
    """Render benchmark results as a Markdown table.

    Rows are grouped by language; columns show loss and latency for each
    strategy side-by-side for easy comparison.

    Args:
        results: List of per-language result dicts produced by
            :func:`run_benchmark`.

    Returns:
        Multi-line Markdown string containing the formatted table.
    """
    # Group by language
    by_lang: dict[str, dict[str, dict]] = defaultdict(dict)
    strategies_seen: set[str] = set()
    for r in results:
        by_lang[r["language"]][r["strategy"]] = r
        strategies_seen.add(r["strategy"])

    ordered_strategies = [s for s in ["a", "b", "baseline"] if s in strategies_seen]
    strategy_labels = {"a": "Strategy A (per-lang)", "b": "Strategy B (generalist)", "baseline": "Baseline (untuned)"}

    # Header
    header_parts = ["| Language | n |"]
    for s in ordered_strategies:
        label = strategy_labels.get(s, s)
        header_parts.append(f" {label} loss | {label} ms/tok |")
    header = "".join(header_parts)
    sep = "| --- | --- |" + " --- | --- |" * len(ordered_strategies)

    rows = [header, sep]
    for lang in LANGUAGES:
        if lang not in by_lang:
            continue
        lang_data = by_lang[lang]
        # Use n from any strategy
        n = next(iter(lang_data.values())).get("n", 0)
        row = f"| {lang} | {n} |"
        for s in ordered_strategies:
            d = lang_data.get(s, {})
            loss = d.get("mean_loss")
            lat = d.get("median_latency_per_token_ms")
            loss_str = f"{loss:.4f}" if loss is not None else "—"
            lat_str = f"{lat:.1f}" if lat is not None else "—"
            row += f" {loss_str} | {lat_str} |"
        rows.append(row)

    return "\n".join(rows)


def save_results(results: list[dict], output_dir: str) -> tuple[Path, Path]:
    """Persist benchmark results to JSON and Markdown files.

    Files are named ``benchmark_{timestamp}.json`` and
    ``benchmark_{timestamp}.md`` where *timestamp* is UTC in
    ``YYYYMMDD_HHMMSS`` format.

    Args:
        results: List of per-language per-strategy result dicts.
        output_dir: Directory path (created if absent).

    Returns:
        Tuple of ``(json_path, md_path)`` :class:`~pathlib.Path` objects.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    json_path = out / f"benchmark_{ts}.json"
    md_path = out / f"benchmark_{ts}.md"

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_examples_evaluated": sum(r["n"] for r in results),
        "results": results,
    }
    json_path.write_text(json.dumps(summary, indent=2))

    md_content = "\n".join([
        f"# Benchmark Results — {ts}",
        "",
        f"Generated: {summary['generated_at']}  ",
        f"Total examples evaluated: {summary['total_examples_evaluated']}",
        "",
        "## Summary Table",
        "",
        _markdown_table(results),
        "",
        "## Notes",
        "",
        "- **Loss** is a token-overlap proxy (1 − F1). Lower = better.",
        "  Real validation perplexity requires logprobs (not available without extra config).",
        "- **ms/tok** is estimated median latency per token (total latency / approx token count).",
        "- Strategy A: per-language Modal endpoints (`gemma-e4b-{lang}-fastapi-app`).",
        "- Strategy B: single generalist Modal endpoint (`gemma-e4b-codealchemy-fastapi-app`).",
        "- Baseline: untuned Gemma 4 E4B via local Ollama.",
    ])
    md_path.write_text(md_content)

    return json_path, md_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: run full benchmark and save results.

    Loads the held-out test set, iterates over all requested languages
    and strategies, collects metrics via :func:`run_benchmark`, and
    writes JSON + Markdown result files via :func:`save_results`.
    """
    args = parse_args()

    print("=" * 60)
    print("benchmark.py — Strategy A vs B vs Baseline")
    print("=" * 60)

    # Load test data
    print(f"\nLoading test set: {args.test_set}")
    records = load_test_set(args.test_set)
    print(f"  Loaded {len(records)} records.")

    # Annotate with inferred language for reporting
    lang_counts: dict[str, int] = defaultdict(int)
    for r in records:
        lang_counts[infer_language(r)] += 1
    print("  Language distribution (inferred):")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"    {lang:12s} {count:4d}")

    # Validate Modal username for non-baseline strategies
    needs_modal = any(s in args.strategies for s in ("a", "b"))
    if needs_modal and not args.username:
        print(
            "\nWARNING: --username not set. Modal endpoint URLs will be malformed.\n"
            "  Use: python benchmark.py --username <your-modal-username>\n"
            "  Proceeding anyway — Modal calls will fail gracefully.\n"
        )

    if args.dry_run:
        print("\nDRY-RUN mode: model calls skipped, synthetic metrics used.")

    # Run evaluations
    all_results: list[dict] = []
    total = len(args.languages) * len(args.strategies)
    done = 0

    for lang in args.languages:
        subset = filter_by_language(records, lang)
        n_avail = len(subset)
        for strategy in args.strategies:
            done += 1
            label = {"a": "Strategy A", "b": "Strategy B", "baseline": "Baseline"}.get(strategy, strategy)
            print(f"\n[{done}/{total}] {label} | {lang} | {n_avail} examples available", flush=True)

            result = run_benchmark(
                strategy=strategy,
                language=lang,
                records=records,
                username=args.username,
                max_per_lang=args.max_per_lang,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
            all_results.append(result)

            n = result["n"]
            loss = result.get("mean_loss")
            lat = result.get("median_latency_per_token_ms")
            errs = len(result.get("errors", []))
            print(
                f"  n={n}  loss={loss if loss is not None else '—'}  "
                f"lat={lat if lat is not None else '—'} ms/tok  errors={errs}"
            )

    # Save
    json_path, md_path = save_results(all_results, args.output_dir)
    print("\n" + "=" * 60)
    print("Results written:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
