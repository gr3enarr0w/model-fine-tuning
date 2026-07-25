#!/usr/bin/env python3
"""
download_from_modal.py — Download trained adapters from Modal volume.

  python download_from_modal.py --model laguna        # Laguna adapter
  python download_from_modal.py --model gemma-python  # Python specialist
  python download_from_modal.py --all                 # all adapters
"""
import argparse
import subprocess
import sys
from pathlib import Path

VOLUME = "model-fine-tuning-vol"
LOCAL_OUTPUTS = Path("outputs")

ADAPTER_PATHS: dict[str, str] = {
    "laguna": "/outputs/laguna-codealchemy-adapter/final-adapter",
    **{
        f"gemma-{lang}": f"/outputs/gemma-e4b-{lang}/final-adapter"
        for lang in [
            "python", "javascript", "go", "rust", "sql", "markdown",
            "typescript", "shell", "java", "php", "c", "cpp", "csharp",
            "ruby", "swift",
        ]
    },
    "gemma-generalist": "/outputs/gemma-e4b-generalist/final-adapter",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for download configuration."""
    parser = argparse.ArgumentParser(
        description="Download trained adapters from Modal volume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        metavar="NAME",
        choices=sorted(ADAPTER_PATHS),
        help=(
            "Name of the adapter to download. "
            "Choices: " + ", ".join(sorted(ADAPTER_PATHS))
        ),
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download all known adapters.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--volume",
        default=VOLUME,
        metavar="NAME",
        help=f"Modal volume name (default: {VOLUME}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LOCAL_OUTPUTS,
        metavar="DIR",
        help=f"Local directory to write adapters into (default: {LOCAL_OUTPUTS}).",
    )
    return parser.parse_args()


def download_adapter(
    model_name: str,
    remote_path: str,
    output_dir: Path,
    volume: str,
    dry_run: bool = False,
) -> bool:
    """Download a named adapter from Modal volume. Returns True on success."""
    # Mirror the remote path structure under output_dir
    # e.g. /outputs/laguna-codealchemy-adapter/final-adapter
    #   -> outputs/laguna-codealchemy-adapter/final-adapter/
    local_dest = output_dir / Path(remote_path.lstrip("/")).parent
    local_dest.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"  [DRY]  {remote_path} -> {local_dest}/", flush=True)
        return True

    print(f"  [DOWN] {model_name}: {remote_path} -> {local_dest}/", flush=True)
    cmd = ["modal", "volume", "get", volume, remote_path, str(local_dest)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            print(f"         ERROR: {err}", file=sys.stderr)
            return False
        return True
    except FileNotFoundError:
        print(
            "ERROR: 'modal' CLI not found. Install with: pip install modal",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    """Download trained adapters from Modal volume to local outputs/."""
    args = parse_args()

    if args.all:
        targets = list(ADAPTER_PATHS.items())
    else:
        targets = [(args.model, ADAPTER_PATHS[args.model])]

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"Modal volume download {mode}← {args.volume}")
    print(f"  {len(targets)} adapter(s) selected\n")

    success_count = 0
    fail_count = 0

    for model_name, remote_path in targets:
        ok = download_adapter(
            model_name=model_name,
            remote_path=remote_path,
            output_dir=args.output_dir,
            volume=args.volume,
            dry_run=args.dry_run,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print()
    print(f"Done: {success_count} succeeded, {fail_count} failed.")

    if fail_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
