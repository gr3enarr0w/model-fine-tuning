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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training strategy and configuration.

    Returns:
        Parsed argument namespace containing strategy, language, config path,
        data directory, and output directory.

    Raises:
        SystemExit: If required arguments are missing or invalid.
    """
    p = argparse.ArgumentParser(
        description="Fine-tune Gemma 4 E4B with per-language or generalist strategy."
    )
    p.add_argument(
        "--strategy",
        choices=["per-language", "generalist"],
        required=True,
        help="Training strategy: per-language specialization or generalist.",
    )
    p.add_argument(
        "--language",
        choices=LANGUAGES,
        help="Target language. Required when --strategy is per-language.",
    )
    p.add_argument(
        "--config",
        default="configs/lora_gemma.yaml",
        help="Path to LoRA YAML config file.",
    )
    p.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing formatted JSONL training data.",
    )
    p.add_argument(
        "--output-dir",
        default="outputs",
        help="Root directory for saving trained adapter weights.",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    """Load and parse a YAML configuration file.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve training data path and output directory based on strategy.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Tuple of (data_path, output_path) as Path objects.
    """
    if args.strategy == "per-language":
        data_path = Path(args.data_dir) / f"train_{args.language}.jsonl"
        output_tag = f"gemma-e4b-{args.language}"
    else:
        data_path = Path(args.data_dir) / "train.jsonl"
        output_tag = "gemma-e4b-generalist"

    output_path = Path(args.output_dir) / output_tag
    output_path.mkdir(parents=True, exist_ok=True)
    return data_path, output_path


def main() -> None:
    """Entry point: validate arguments, load config, and launch training.

    Validates that --language is provided when strategy is per-language,
    resolves data and output paths, logs resolved parameters, and
    delegates to the training implementation.

    Raises:
        ValueError: If --language is omitted for per-language strategy.
    """
    args = parse_args()

    if args.strategy == "per-language" and not args.language:
        raise ValueError("--language is required when --strategy is per-language")

    cfg = load_config(args.config)
    data_path, output_path = resolve_paths(args)

    print(f"Strategy  : {args.strategy}")
    print(f"Language  : {args.language or 'all'}")
    print(f"Data      : {data_path}")
    print(f"Output    : {output_path}")
    print(f"Config    : {cfg}")
    print("Training not yet implemented — scaffold only")


if __name__ == "__main__":
    main()
