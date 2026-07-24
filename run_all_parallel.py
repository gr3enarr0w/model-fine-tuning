#!/usr/bin/env python3
"""
run_all_parallel.py — Orchestrate all training runs within a $30 Modal credit budget.

Strategy
--------
  Laguna XS 2.1  (A100-40GB, ~$20 est.)  — runs in parallel with Batch 1
  Batch 1: Python, JavaScript, Go, Rust   — highest-value languages
  Batch 2: SQL, Markdown, TypeScript, Shell
  Batch 3: Java, PHP, C, C++, C#, Ruby, Swift
  Generalist                              — last (benefits from all prior data)

Budget logic
------------
  - Tracks estimated spend (Laguna $20 + $2 per Gemma model conservative)
  - Stops launching new batches when remaining budget < $2
  - Uses Modal .spawn() for non-blocking parallel execution within each batch
  - Waits for each batch to finish before launching the next

Usage
-----
  modal run run_all_parallel.py                     # full run
  modal run run_all_parallel.py --dry-run           # print plan, don't launch
  modal run run_all_parallel.py --max-steps 50      # smoke test all jobs
  modal run run_all_parallel.py --limit 500         # cap examples per job
"""
from __future__ import annotations

import time

import modal

BUDGET = 30.0
LAGUNA_COST_ESTIMATE = 20.0   # A100-40GB at $2.80/hr × ~7h
GEMMA_COST_PER_MODEL = 2.0    # A10G at $1.10/hr × ~2h, conservative

# Priority-ordered language batches.  Generalist is intentionally last.
BATCHES: list[list[str | None]] = [
    # Batch 1 — highest-ROI languages; runs parallel with Laguna
    ["python", "javascript", "go", "rust"],
    # Batch 2
    ["sql", "markdown", "typescript", "shell"],
    # Batch 3
    ["java", "php", "c", "cpp", "csharp", "ruby", "swift"],
    # Generalist (None signals generalist strategy)
    [None],
]

# Flat priority list for display and sequential budget checks
LANGUAGE_PRIORITY: list[str] = [
    # Batch 1
    "python", "javascript", "go", "rust",
    # Batch 2
    "sql", "markdown", "typescript", "shell",
    # Batch 3
    "java", "php", "c", "cpp", "csharp", "ruby", "swift",
    # Generalist
    "generalist",
]


def _label(language: str | None) -> str:
    return language if language is not None else "generalist"


def _budget_ok(spent: float, cost: float) -> bool:
    return (BUDGET - spent - cost) >= 0


def main(
    dry_run: bool = False,
    max_steps: int = -1,
    limit: int = 0,
) -> None:
    """
    Orchestrate all training runs with budget tracking.

    Args:
        dry_run: Print the execution plan without launching any jobs.
        max_steps: Override max training steps in every job (smoke test).
        limit: Cap training examples per job (smoke test).
    """
    spent = 0.0
    remaining = BUDGET - spent

    print("=" * 60)
    print("Modal Training Orchestrator — $30 Budget")
    print("=" * 60)
    print(f"  Budget             : ${BUDGET:.0f}")
    print(f"  Laguna estimate    : ~${LAGUNA_COST_ESTIMATE:.0f}  (A100-40GB)")
    print(f"  Gemma per model    : ~${GEMMA_COST_PER_MODEL:.0f}  (A10G, conservative)")
    max_gemma = int((BUDGET - LAGUNA_COST_ESTIMATE) / GEMMA_COST_PER_MODEL)
    print(f"  Max Gemma models   : ~{max_gemma}")
    if max_steps > 0:
        print(f"  [Smoke] max-steps  : {max_steps}")
    if limit > 0:
        print(f"  [Smoke] limit      : {limit} examples")
    print()

    if dry_run:
        print("[DRY RUN] Execution plan (no jobs will be launched):\n")
        _print_plan()
        return

    # ------------------------------------------------------------------
    # Import the Modal functions from their respective modules
    # ------------------------------------------------------------------
    from train import train as laguna_train          # noqa: F401  — A100 job
    from train_gemma import train_modal as gemma_train  # noqa: F401  — A10G job

    # ------------------------------------------------------------------
    # Launch Laguna + Batch 1 simultaneously
    # ------------------------------------------------------------------
    if not _budget_ok(spent, LAGUNA_COST_ESTIMATE):
        print(f"[Budget] Insufficient funds for Laguna (${LAGUNA_COST_ESTIMATE:.0f}). Skipping.")
        laguna_handle = None
    else:
        print(f"[Launch] Laguna XS 2.1 on A100  (est. ~${LAGUNA_COST_ESTIMATE:.0f})")
        laguna_handle = laguna_train.spawn(
            limit=limit if limit > 0 else None,
            max_steps_override=max_steps,
        )
        spent += LAGUNA_COST_ESTIMATE
        print(f"         Spawned. Estimated spend so far: ~${spent:.0f}")

    # Batch 1 runs in parallel with Laguna
    batch_handles: dict[str, object] = {}

    print(f"\n[Batch 1] Python, JavaScript, Go, Rust — parallel with Laguna")
    for lang in BATCHES[0]:
        label = _label(lang)
        if not _budget_ok(spent, GEMMA_COST_PER_MODEL):
            print(f"  [Budget] Skipping {label} — remaining budget < ${GEMMA_COST_PER_MODEL:.0f}")
            continue
        print(f"  [Launch] gemma-e4b-{label} on A10G  (est. ~${GEMMA_COST_PER_MODEL:.0f})")
        handle = gemma_train.spawn(
            language=lang,
            strategy="per-language" if lang else "generalist",
            limit=limit if limit > 0 else None,
            max_steps_override=max_steps,
        )
        batch_handles[label] = handle
        spent += GEMMA_COST_PER_MODEL
        print(f"         Spawned. Estimated spend: ~${spent:.0f}")

    # Wait for Laguna
    if laguna_handle is not None:
        print("\n[Wait] Waiting for Laguna XS 2.1 to finish...")
        try:
            laguna_handle.get()
            print("[Done] Laguna XS 2.1 training complete.")
        except Exception as exc:
            print(f"[Error] Laguna failed: {exc}")

    # Wait for Batch 1
    _await_batch(batch_handles, batch_name="Batch 1")

    # ------------------------------------------------------------------
    # Remaining batches — sequential with budget checks
    # ------------------------------------------------------------------
    for batch_num, batch in enumerate(BATCHES[1:], start=2):
        batch_labels = [_label(l) for l in batch]
        print(f"\n[Batch {batch_num}] {', '.join(batch_labels)}")
        print(f"  Remaining budget: ~${BUDGET - spent:.0f}")

        batch_handles = {}
        for lang in batch:
            label = _label(lang)
            if not _budget_ok(spent, GEMMA_COST_PER_MODEL):
                print(f"  [Budget] Stopping — remaining budget < ${GEMMA_COST_PER_MODEL:.0f}. "
                      f"Skipping {label} and remaining languages.")
                break
            strategy = "per-language" if lang else "generalist"
            print(f"  [Launch] gemma-e4b-{label} on A10G  (est. ~${GEMMA_COST_PER_MODEL:.0f})")
            handle = gemma_train.spawn(
                language=lang,
                strategy=strategy,
                limit=limit if limit > 0 else None,
                max_steps_override=max_steps,
            )
            batch_handles[label] = handle
            spent += GEMMA_COST_PER_MODEL
            print(f"           Spawned. Estimated spend: ~${spent:.0f}")

        _await_batch(batch_handles, batch_name=f"Batch {batch_num}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All jobs finished.")
    print(f"  Estimated total spend : ~${spent:.0f}")
    print(f"  Budget remaining      : ~${BUDGET - spent:.0f}")
    print()
    print("Retrieve adapters with:")
    print("  modal volume get model-fine-tuning-vol /outputs/gemma-e4b-python/final-adapter ./adapters/python")
    print("  modal volume get laguna-codealchemy-vol /outputs/final-adapter ./adapters/laguna")
    print("=" * 60)


def _await_batch(handles: dict[str, object], batch_name: str) -> None:
    """Block until all handles in the batch complete, reporting results."""
    if not handles:
        return
    print(f"\n[Wait] Waiting for {batch_name} ({len(handles)} job(s))...")
    for label, handle in handles.items():
        try:
            handle.get()
            print(f"  [Done] gemma-e4b-{label}")
        except Exception as exc:
            print(f"  [Error] gemma-e4b-{label} failed: {exc}")
    print(f"[Done] {batch_name} complete.")


def _print_plan() -> None:
    """Print the full execution plan for --dry-run mode."""
    spent = 0.0
    print(f"  Phase 0+1 (parallel):")
    print(f"    Laguna XS 2.1      A100-40GB  ~${LAGUNA_COST_ESTIMATE:.0f}")
    spent += LAGUNA_COST_ESTIMATE
    for lang in BATCHES[0]:
        label = _label(lang)
        ok = _budget_ok(spent, GEMMA_COST_PER_MODEL)
        status = "LAUNCH" if ok else "SKIP (budget)"
        print(f"    gemma-e4b-{label:<12} A10G   ~${GEMMA_COST_PER_MODEL:.0f}  [{status}]")
        if ok:
            spent += GEMMA_COST_PER_MODEL
    for batch_num, batch in enumerate(BATCHES[1:], start=2):
        print(f"\n  Batch {batch_num} (sequential):")
        for lang in batch:
            label = _label(lang)
            ok = _budget_ok(spent, GEMMA_COST_PER_MODEL)
            status = "LAUNCH" if ok else "SKIP (budget)"
            print(f"    gemma-e4b-{label:<12} A10G   ~${GEMMA_COST_PER_MODEL:.0f}  [{status}]")
            if ok:
                spent += GEMMA_COST_PER_MODEL
    print(f"\n  Estimated total: ~${spent:.0f} / ${BUDGET:.0f} budget")


# ---------------------------------------------------------------------------
# Modal local entrypoint
# ---------------------------------------------------------------------------

app = modal.App("gemma-orchestrator")


@app.local_entrypoint()
def entrypoint(
    dry_run: bool = False,
    max_steps: int = -1,
    limit: int = 0,
) -> None:
    """
    Orchestrate all Gemma E4B + Laguna training runs within a $30 budget.

    Run with:
        modal run run_all_parallel.py                      # full run
        modal run run_all_parallel.py --dry-run            # print plan only
        modal run run_all_parallel.py --max-steps 50       # smoke test
        modal run run_all_parallel.py --limit 500          # cap examples
    """
    main(dry_run=dry_run, max_steps=max_steps, limit=limit)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Orchestrate all Modal training jobs.")
    p.add_argument("--dry-run", action="store_true", help="Print plan without launching.")
    p.add_argument("--max-steps", type=int, default=-1, help="Smoke test: stop after N steps.")
    p.add_argument("--limit", type=int, default=0, help="Smoke test: cap examples per job.")
    args = p.parse_args()
    main(dry_run=args.dry_run, max_steps=args.max_steps, limit=args.limit)
