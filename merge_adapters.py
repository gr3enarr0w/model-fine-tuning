#!/usr/bin/env python3
"""
merge_adapters.py — Merge LoRA adapter weights into base model.

Required before pushing to Ollama (GGUF conversion needs merged weights).

  python merge_adapters.py --model laguna --output merged/laguna
  python merge_adapters.py --model gemma-python --output merged/gemma-python
"""
import argparse
from pathlib import Path

ADAPTER_REGISTRY = {
    "laguna": {
        "base": "poolside/Laguna-XS-2.1",
        "adapter": "outputs/laguna-codealchemy-adapter/final-adapter",
    },
    **{f"gemma-{lang}": {
        "base": "google/gemma-4-e4b-it",
        "adapter": f"outputs/gemma-e4b-{lang}/final-adapter",
    } for lang in ["python","javascript","go","rust","sql","markdown",
                   "typescript","shell","java","php","c","cpp","csharp","ruby","swift"]},
    "gemma-generalist": {
        "base": "google/gemma-4-e4b-it",
        "adapter": "outputs/gemma-e4b-generalist/final-adapter",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments: which model to merge and output path."""
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into its base model for deployment."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(ADAPTER_REGISTRY.keys()),
        help="Which adapter to merge (e.g. laguna, gemma-python).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Local directory to save the merged model (e.g. merged/laguna).",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for the merged weights (default: float16).",
    )
    return parser.parse_args()


def merge(base_model_id: str, adapter_path: str, output_path: str, dtype: str = "float16") -> None:
    """Load base model + LoRA adapter, merge, save as HF model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.save_pretrained(output_path)

    print(f"Done. Merged model saved to {output_path}")


def main() -> None:
    """Entry point: validate paths, run merge."""
    args = parse_args()

    entry = ADAPTER_REGISTRY[args.model]
    base_model_id = entry["base"]
    adapter_path = entry["adapter"]

    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_path}\n"
            f"Run training first, or check that the adapter path is correct."
        )

    merge(
        base_model_id=base_model_id,
        adapter_path=str(adapter_dir),
        output_path=args.output,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
