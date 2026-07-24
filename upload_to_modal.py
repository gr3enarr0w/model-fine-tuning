#!/usr/bin/env python3
"""
upload_to_modal.py — Upload training data and configs to Modal volume.

Run this before any training job:
  python upload_to_modal.py                    # upload data + configs
  python upload_to_modal.py --data-only        # skip configs
  python upload_to_modal.py --dry-run          # show what would upload
"""
import argparse
import subprocess
import sys
from pathlib import Path

VOLUME = "model-fine-tuning-vol"

UPLOADS = [
    ("data/train.jsonl",        "/data/train.jsonl"),
    ("data/val.jsonl",          "/data/val.jsonl"),
    ("data/test.jsonl",         "/data/test.jsonl"),
    ("configs/qlora.yaml",      "/configs/qlora.yaml"),
    ("configs/lora_gemma.yaml", "/configs/lora_gemma.yaml"),
]

DATA_FILES = {"data/train.jsonl", "data/val.jsonl", "data/test.jsonl"}
CONFIG_FILES = {"configs/qlora.yaml", "configs/lora_gemma.yaml"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for upload configuration."""
    parser = argparse.ArgumentParser(
        description="Upload training data and configs to Modal volume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Upload training data files only; skip config files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading.",
    )
    parser.add_argument(
        "--volume",
        default=VOLUME,
        metavar="NAME",
        help=f"Modal volume name (default: {VOLUME}).",
    )
    return parser.parse_args()


def upload_file(local: str, remote: str, volume: str, dry_run: bool = False) -> bool:
    """Upload a single file to Modal volume. Returns True on success."""
    local_path = Path(local)

    if not local_path.exists():
        print(f"  [SKIP] {local} — file not found", flush=True)
        return True  # non-fatal; skip missing optional files

    if dry_run:
        print(f"  [DRY]  {local} -> {remote}", flush=True)
        return True

    print(f"  [UP]   {local} -> {remote}", flush=True)
    cmd = ["modal", "volume", "put", volume, str(local_path), remote]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"         ERROR: {result.stderr.strip()}", file=sys.stderr)
            return False
        return True
    except FileNotFoundError:
        print(
            "ERROR: 'modal' CLI not found. Install with: pip install modal",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    """Upload all training artifacts to Modal volume."""
    args = parse_args()

    # Filter upload list based on flags
    uploads = [
        (local, remote)
        for local, remote in UPLOADS
        if not args.data_only or local in DATA_FILES
    ]

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"Modal volume upload {mode}→ {args.volume}")
    print(f"  {len(uploads)} file(s) selected\n")

    success_count = 0
    fail_count = 0

    for local, remote in uploads:
        ok = upload_file(local, remote, volume=args.volume, dry_run=args.dry_run)
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
