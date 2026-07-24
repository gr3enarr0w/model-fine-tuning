#!/usr/bin/env python3
"""
run_all_parallel.py — Orchestrate Phase 1 training runs within a $30 Modal credit budget.

Strategy
--------
  Phase 1 (default, --phase 1):
    Laguna XS 2.1  (A100-40GB, ~$20 est.)  — runs in parallel with Phase 1 Gemma models
    Python, Rust, Generalist               — hypothesis test on A10G

  Phase 2 (manual trigger, --phase 2):
    Remaining languages: JavaScript, Go, SQL, Markdown, TypeScript, Shell,
    Java, PHP, C, C++, C#, Ruby, Swift
    Only run if Phase 1 benchmark shows specialization beats generalist.

Budget logic
------------
  - Tracks estimated spend (Laguna $20 + $2 per Gemma model conservative)
  - Stops launching new batches when remaining budget < $2
  - Uses Modal .spawn() for non-blocking parallel execution within each batch
  - Waits for each batch to finish before launching the next

Usage
-----
  modal run run_all_parallel.py                     # Phase 1: python+rust+generalist
  modal run run_all_parallel.py --phase 2           # Phase 2: remaining languages
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

# Phase 1: Hypothesis test — does per-language specialization beat the generalist?
# Train 2 languages + 1 generalist. Benchmark all 3.
# If specialization wins → run Phase 2 (remaining languages).
# If not → generalist is sufficient. No Phase 2 needed.
# Run Phase 2 manually: modal run run_all_parallel.py --phase 2
PHASE_1_LANGUAGES = ["python", "rust", "generalist"]

# Phase 2: Remaining languages — only run if benchmark shows specialization beats generalist.
PHASE_2_LANGUAGES = [
    "javascript", "go", "sql", "markdown", "typescript", "shell",
    "java", "php", "c", "cpp", "csharp", "ruby", "swift",
]

# Flat priority list for display and sequential budget checks
LANGUAGE_PRIORITY: list[str] = ["python", "rust", "generalist"]


def _label(language: str | None) -> str:
    return language if language is not None else "generalist"


def _budget_ok(spent: float, cost: float) -> bool:
    return (BUDGET - spent - cost) >= 0


def main(
    dry_run: bool = False,
    max_steps: int = -1,
    limit: int = 0,
    phase: int = 1,
) -> None:
    """
    Orchestrate training runs with budget tracking.

    Args:
        dry_run: Print the execution plan without launching any jobs.
        max_steps: Override max training steps in every job (smoke test).
        limit: Cap training examples per job (smoke test).
        phase: 1 = python+rust+generalist (default hypothesis test);
               2 = remaining languages (run only if specialization won benchmark).
    """
    spent = 0.0

    print("=" * 60)
    print(f"Modal Training Orchestrator — Phase {phase} — $30 Budget")
    print("=" * 60)
    print(f"  Budget             : ${BUDGET:.0f}")
    print(f"  Laguna estimate    : ~${LAGUNA_COST_ESTIMATE:.0f}  (A100-40GB)")
    print(f"  Gemma per model    : ~${GEMMA_COST_PER_MODEL:.0f}  (A10G, conservative)")
    max_gemma = int((BUDGET - LAGUNA_COST_ESTIMATE) / GEMMA_COST_PER_MODEL)
    print(f"  Max Gemma models   : ~{max_gemma}")
    if phase == 1:
        print(f"  Phase 1 models     : {', '.join(PHASE_1_LANGUAGES)}")
    else:
        print(f"  Phase 2 models     : {', '.join(PHASE_2_LANGUAGES)}")
    if max_steps > 0:
        print(f"  [Smoke] max-steps  : {max_steps}")
    if limit > 0:
        print(f"  [Smoke] limit      : {limit} examples")
    print()

    if dry_run:
        print("[DRY RUN] Execution plan (no jobs will be launched):\n")
        _print_plan(phase=phase)
        return

    # ------------------------------------------------------------------
    # Import the Modal functions from their respective modules
    # ------------------------------------------------------------------
    from train import train as laguna_train          # noqa: F401  — A100 job
    from train_gemma import train_modal as gemma_train  # noqa: F401  — A10G job

    # ------------------------------------------------------------------
    # Phase 1: Launch Laguna + Phase 1 Gemma models simultaneously
    # Phase 2: Skip Laguna (already done), just run remaining languages
    # ------------------------------------------------------------------
    laguna_handle = None
    if phase == 1:
        if not _budget_ok(spent, LAGUNA_COST_ESTIMATE):
            print(f"[Budget] Insufficient funds for Laguna (${LAGUNA_COST_ESTIMATE:.0f}). Skipping.")
        else:
            print(f"[Launch] Laguna XS 2.1 on A100  (est. ~${LAGUNA_COST_ESTIMATE:.0f})")
            laguna_handle = laguna_train.spawn(
                limit=limit if limit > 0 else None,
                max_steps_override=max_steps,
            )
            spent += LAGUNA_COST_ESTIMATE
            print(f"         Spawned. Estimated spend so far: ~${spent:.0f}")

    # Select language list for this phase
    if phase == 1:
        phase_langs = PHASE_1_LANGUAGES
        phase_label = "Phase 1 (python, rust, generalist) — parallel with Laguna"
    else:
        phase_langs = PHASE_2_LANGUAGES
        phase_label = "Phase 2 (remaining languages) — post-benchmark run"

    batch_handles: dict[str, object] = {}
    print(f"\n[{phase_label}]")

    for lang_key in phase_langs:
        # "generalist" maps to language=None in the training function
        lang = None if lang_key == "generalist" else lang_key
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

    # Wait for Laguna (Phase 1 only)
    if laguna_handle is not None:
        print("\n[Wait] Waiting for Laguna XS 2.1 to finish...")
        try:
            laguna_handle.get()
            print("[Done] Laguna XS 2.1 training complete.")
        except Exception as exc:
            print(f"[Error] Laguna failed: {exc}")

    # Wait for all Gemma models in this phase
    _await_batch(batch_handles, batch_name=f"Phase {phase}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Phase {phase} complete.")
    print(f"  Estimated total spend : ~${spent:.0f}")
    print(f"  Budget remaining      : ~${BUDGET - spent:.0f}")
    print()
    if phase == 1:
        print("Next steps:")
        print("  1. Benchmark python, rust, and generalist adapters.")
        print("  2. If specialization wins → run Phase 2:")
        print("       modal run run_all_parallel.py --phase 2")
        print("  3. If generalist wins → no Phase 2 needed.")
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


def _print_plan(phase: int = 1) -> None:
    """Print the full execution plan for --dry-run mode."""
    spent = 0.0
    if phase == 1:
        print(f"  Phase 1 (parallel):")
        print(f"    Laguna XS 2.1      A100-40GB  ~${LAGUNA_COST_ESTIMATE:.0f}")
        spent += LAGUNA_COST_ESTIMATE
        for lang_key in PHASE_1_LANGUAGES:
            lang = None if lang_key == "generalist" else lang_key
            label = _label(lang)
            ok = _budget_ok(spent, GEMMA_COST_PER_MODEL)
            status = "LAUNCH" if ok else "SKIP (budget)"
            print(f"    gemma-e4b-{label:<12} A10G   ~${GEMMA_COST_PER_MODEL:.0f}  [{status}]")
            if ok:
                spent += GEMMA_COST_PER_MODEL
        print(f"\n  Estimated total: ~${spent:.0f} / ${BUDGET:.0f} budget")
        print(f"\n  Phase 2 (manual, after benchmark):")
        for lang_key in PHASE_2_LANGUAGES:
            print(f"    gemma-e4b-{lang_key:<12} A10G   ~${GEMMA_COST_PER_MODEL:.0f}  [PENDING benchmark]")
    else:
        print(f"  Phase 2 (remaining languages):")
        for lang_key in PHASE_2_LANGUAGES:
            lang = None if lang_key == "generalist" else lang_key
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
    phase: int = 1,
) -> None:
    """
    Orchestrate Gemma E4B + Laguna training runs within a $30 budget.

    Run with:
        modal run run_all_parallel.py                      # Phase 1 (python+rust+generalist)
        modal run run_all_parallel.py --phase 2            # Phase 2 (remaining languages)
        modal run run_all_parallel.py --dry-run            # print plan only
        modal run run_all_parallel.py --max-steps 50       # smoke test
        modal run run_all_parallel.py --limit 500          # cap examples
    """
    main(dry_run=dry_run, max_steps=max_steps, limit=limit, phase=phase)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Orchestrate Modal training jobs by phase.")
    p.add_argument("--dry-run", action="store_true", help="Print plan without launching.")
    p.add_argument("--max-steps", type=int, default=-1, help="Smoke test: stop after N steps.")
    p.add_argument("--limit", type=int, default=0, help="Smoke test: cap examples per job.")
    p.add_argument(
        "--phase", type=int, default=1, choices=[1, 2],
        help=(
            "Phase 1 (default): python+rust+generalist hypothesis test. "
            "Phase 2: remaining languages — run only if specialization beat generalist."
        ),
    )
    args = p.parse_args()
    main(dry_run=args.dry_run, max_steps=args.max_steps, limit=args.limit, phase=args.phase)
