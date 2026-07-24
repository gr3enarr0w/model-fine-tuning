"""
Gemma 4 E4B LoRA fine-tuning — two strategies:
  Strategy A: per-language specialization (one model per language)
  Strategy B: generalist (single model, all languages)

Run locally on M2 Pro or on Modal for faster iteration.

Usage:
  python train_gemma.py --strategy per-language --language python
  python train_gemma.py --strategy generalist
"""
import argparse
import yaml
from pathlib import Path

LANGUAGES = ["python", "go", "typescript", "java"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["per-language", "generalist"], required=True)
    p.add_argument("--language", choices=LANGUAGES, help="Required for per-language strategy")
    p.add_argument("--config", default="configs/lora_gemma.yaml")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    if args.strategy == "per-language" and not args.language:
        raise ValueError("--language required for per-language strategy")

    cfg = load_config(args.config)
    output_tag = f"gemma-e4b-{args.language}" if args.strategy == "per-language" else "gemma-e4b-generalist"
    output_path = Path(args.output_dir) / output_tag
    output_path.mkdir(parents=True, exist_ok=True)

    data_path = (
        Path(args.data_dir) / f"train_{args.language}.jsonl"
        if args.strategy == "per-language"
        else Path(args.data_dir) / "train.jsonl"
    )

    print(f"Strategy  : {args.strategy}")
    print(f"Language  : {args.language or 'all'}")
    print(f"Data      : {data_path}")
    print(f"Output    : {output_path}")
    print(f"Config    : {cfg}")
    print("Training not yet implemented — scaffold only")

if __name__ == "__main__":
    main()
