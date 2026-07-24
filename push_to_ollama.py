#!/usr/bin/env python3
"""
push_to_ollama.py — Convert merged HF model to GGUF and create Ollama modelfile.

Requires: llama.cpp (for conversion) and ollama CLI.

  python push_to_ollama.py --model gemma-python --quantize q5_k_m
"""
import argparse
import subprocess
import shutil
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert software engineer. "
    "Write clean, correct, well-documented code. "
    "If you are unsure, say so rather than guessing."
)

LLAMACPP_CONVERT_SCRIPT = "convert_hf_to_gguf.py"


def parse_args() -> argparse.Namespace:
    """Parse model name, merged path, quantization level, and Ollama tag."""
    parser = argparse.ArgumentParser(
        description="Convert a merged HuggingFace model to GGUF and register with Ollama."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier (e.g. gemma-python, laguna).",
    )
    parser.add_argument(
        "--merged-path",
        default=None,
        help="Path to the merged HF model directory (default: merged/<model>).",
    )
    parser.add_argument(
        "--gguf-dir",
        default="gguf",
        help="Directory to store GGUF output files (default: gguf/).",
    )
    parser.add_argument(
        "--quantize",
        default="q5_k_m",
        choices=["q4_0", "q4_k_m", "q5_0", "q5_k_m", "q6_k", "q8_0", "f16"],
        help="Quantization format for GGUF (default: q5_k_m).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Ollama model tag (default: <model>:latest).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to embed in the Ollama Modelfile.",
    )
    parser.add_argument(
        "--llamacpp-dir",
        default=None,
        help="Path to llama.cpp directory containing conversion scripts.",
    )
    return parser.parse_args()


def convert_to_gguf(merged_path: str, output_path: str, quantize: str = "q5_k_m", llamacpp_dir: str | None = None) -> str:
    """Convert HuggingFace model to GGUF via llama.cpp convert script."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate the conversion script
    if llamacpp_dir:
        convert_script = Path(llamacpp_dir) / LLAMACPP_CONVERT_SCRIPT
    else:
        # Try to find it on PATH or common locations
        found = shutil.which(LLAMACPP_CONVERT_SCRIPT)
        if found:
            convert_script = Path(found)
        else:
            raise FileNotFoundError(
                f"Could not find {LLAMACPP_CONVERT_SCRIPT}. "
                "Pass --llamacpp-dir pointing to your llama.cpp checkout."
            )

    f16_gguf = output_dir / "model-f16.gguf"

    print(f"Converting {merged_path} to GGUF (f16 pass)...")
    subprocess.run(
        ["python", str(convert_script), merged_path, "--outfile", str(f16_gguf), "--outtype", "f16"],
        check=True,
    )

    if quantize == "f16":
        print(f"Skipping quantization (f16 requested). Output: {f16_gguf}")
        return str(f16_gguf)

    quantized_gguf = output_dir / f"model-{quantize}.gguf"
    print(f"Quantizing to {quantize}...")

    llama_quantize = shutil.which("llama-quantize") or shutil.which("quantize")
    if llamacpp_dir and not llama_quantize:
        llama_quantize = str(Path(llamacpp_dir) / "llama-quantize")

    if not llama_quantize:
        raise FileNotFoundError(
            "Could not find llama-quantize binary. "
            "Build llama.cpp and ensure llama-quantize is on PATH."
        )

    subprocess.run(
        [llama_quantize, str(f16_gguf), str(quantized_gguf), quantize.upper()],
        check=True,
    )

    print(f"Quantized GGUF: {quantized_gguf}")
    return str(quantized_gguf)


def create_modelfile(gguf_path: str, model_name: str, system_prompt: str) -> str:
    """Write an Ollama Modelfile for the converted model."""
    modelfile_path = Path(gguf_path).parent / f"Modelfile.{model_name}"
    content = f"""\
FROM {gguf_path}

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 8192

SYSTEM \"\"\"{system_prompt}\"\"\"
"""
    modelfile_path.write_text(content)
    print(f"Modelfile written: {modelfile_path}")
    return str(modelfile_path)


def register_with_ollama(modelfile_path: str, tag: str) -> bool:
    """Run ollama create to register the model locally."""
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        print("WARNING: ollama not found on PATH. Skipping registration.")
        print(f"To register manually:\n  ollama create {tag} -f {modelfile_path}")
        return False

    print(f"Registering model with Ollama as {tag}...")
    result = subprocess.run(
        [ollama_bin, "create", tag, "-f", modelfile_path],
        check=False,
    )
    if result.returncode == 0:
        print(f"Model registered: {tag}")
        print(f"Test with: ollama run {tag}")
        return True
    else:
        print(f"ollama create exited with code {result.returncode}")
        return False


def main() -> None:
    """Convert, create Modelfile, and register with Ollama."""
    args = parse_args()

    merged_path = args.merged_path or f"merged/{args.model}"
    tag = args.tag or f"{args.model}:latest"
    gguf_output_dir = str(Path(args.gguf_dir) / args.model)

    if not Path(merged_path).exists():
        raise FileNotFoundError(
            f"Merged model not found at: {merged_path}\n"
            "Run merge_adapters.py first."
        )

    gguf_path = convert_to_gguf(
        merged_path=merged_path,
        output_path=gguf_output_dir,
        quantize=args.quantize,
        llamacpp_dir=args.llamacpp_dir,
    )

    modelfile_path = create_modelfile(
        gguf_path=gguf_path,
        model_name=args.model,
        system_prompt=args.system_prompt,
    )

    register_with_ollama(modelfile_path=modelfile_path, tag=tag)


if __name__ == "__main__":
    main()
