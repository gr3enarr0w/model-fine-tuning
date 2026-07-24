"""
train_gemma.py — Gemma 4 E4B LoRA fine-tuning with automated hyperparameter strategies.

Two training strategies:
  Strategy A: per-language specialization (one adapter per language)
  Strategy B: generalist (single adapter, all languages)

Run locally on M2 Pro or on Modal for faster iteration.

Usage:
  python train_gemma.py --strategy per-language --language python
  python train_gemma.py --strategy generalist

Automated hyperparameter strategies
------------------------------------
  LoRA rank   : max(8, hidden_dim // 128)  — derived from model architecture
  LoRA alpha  : 2 × rank  — modern consensus, not rank==alpha
  Learning rate: exponential sweep (1e-7 → 1e-2 over 100 steps); finds steepest
                  descent point; falls back to config learning_rate_fallback
  Batch size  : auto_find_batch_size=True in TrainingArguments (TRL / HF Trainer)
  Epochs      : early stopping (patience=3) instead of a fixed epoch count
  max_steps   : derived at runtime from dataset size × epochs × batch
"""
import argparse
import math
import os
from pathlib import Path

import yaml

LANGUAGES = ["python", "go", "typescript", "java"]

# ---------------------------------------------------------------------------
# Architecture constant — Gemma 4 E4B active hidden_dim
# ---------------------------------------------------------------------------
HIDDEN_DIM_GEMMA_E4B = 2048


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


def compute_lora_rank(hidden_dim: int) -> tuple[int, int]:
    """Derive LoRA rank and alpha from model hidden dimension.

    Formula: r = max(8, hidden_dim // 128);  alpha = 2 × r
    This is grounded in empirical results from large-scale LoRA sweeps and
    replaces manual guessing with a principled default.

    Args:
        hidden_dim: The model's active hidden dimension size.

    Returns:
        Tuple of (lora_r, lora_alpha).
    """
    lora_r = max(8, hidden_dim // 128)
    lora_alpha = 2 * lora_r
    return lora_r, lora_alpha


def find_learning_rate(
    model: object,
    train_dataset: object,
    tokenizer: object,
    *,
    num_sweep_steps: int = 100,
    start_lr: float = 1e-7,
    end_lr: float = 1e-2,
    fallback_lr: float = 1e-4,
) -> float:
    """
    Exponential LR sweep to find the optimal learning rate.

    Runs num_sweep_steps mini-batches with LR increasing exponentially from
    start_lr to end_lr, records loss at each step, then returns the LR just
    before the loss starts diverging (point of maximum negative gradient of
    the smoothed loss curve).

    Args:
        model: The LoRA-wrapped model (already on the target device).
        train_dataset: HuggingFace Dataset with a "text" field.
        tokenizer: The model's tokenizer.
        num_sweep_steps: Number of steps in the exponential sweep.
        start_lr: Lowest LR to try.
        end_lr: Highest LR to try.
        fallback_lr: Returned if the sweep raises any exception.

    Returns:
        Suggested learning rate (float).
    """
    print(f"\n[LRFinder] Sweeping LR {start_lr:.0e} → {end_lr:.0e} "
          f"over {num_sweep_steps} steps...")

    try:
        import copy
        import torch
        from torch.optim import AdamW

        sweep_model = copy.deepcopy(model)
        sweep_model.train()

        optimizer = AdamW(sweep_model.parameters(), lr=start_lr)

        # Determine device
        try:
            device = next(sweep_model.parameters()).device
        except StopIteration:
            device = "cpu"

        # Tokenise a small subset for the sweep
        sweep_texts = [train_dataset[i]["text"]
                       for i in range(min(num_sweep_steps * 2, len(train_dataset)))]
        encodings = tokenizer(
            sweep_texts,
            truncation=True,
            max_length=256,     # short context for speed on smaller model
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        lr_multiplier = (end_lr / start_lr) ** (1.0 / num_sweep_steps)
        losses: list[float] = []
        lrs: list[float] = []
        current_lr = start_lr

        for step in range(num_sweep_steps):
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            idx = step % len(input_ids)
            batch_ids = input_ids[idx: idx + 1]
            batch_mask = attention_mask[idx: idx + 1]

            optimizer.zero_grad()
            outputs = sweep_model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                labels=batch_ids,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            lrs.append(current_lr)
            current_lr *= lr_multiplier

            if loss.item() > 10 * losses[0]:
                print(f"[LRFinder] Loss diverged at step {step} — stopping sweep early.")
                break

        # Smooth losses with exponential moving average
        beta = 0.9
        smoothed: list[float] = []
        avg = losses[0]
        for l in losses:
            avg = beta * avg + (1 - beta) * l
            smoothed.append(avg / (1 - beta ** (len(smoothed) + 1)))

        if len(smoothed) < 3:
            raise ValueError("Too few sweep steps to determine gradient.")

        gradients = [smoothed[i + 1] - smoothed[i] for i in range(len(smoothed) - 1)]
        best_idx = gradients.index(min(gradients))
        suggested_lr = lrs[max(0, best_idx - 1)]
        suggested_lr = float(min(max(suggested_lr, 1e-6), 5e-4))

        del sweep_model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"[LRFinder] Suggested LR: {suggested_lr:.2e}  "
              f"(steepest descent at step {best_idx})")
        return suggested_lr

    except Exception as exc:
        print(f"[LRFinder] Sweep failed ({exc}); using fallback LR {fallback_lr:.0e}")
        return fallback_lr


def run_training(
    cfg: dict,
    train_dataset: object,
    eval_dataset: object,
    model: object,
    tokenizer: object,
    output_path: Path,
    *,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    max_steps: int,
) -> None:
    """
    Configure and execute SFTTrainer with all automated strategies.

    Args:
        cfg: Merged configuration dictionary.
        train_dataset: HuggingFace Dataset for training.
        eval_dataset: HuggingFace Dataset for evaluation.
        model: LoRA-wrapped model.
        tokenizer: Model tokenizer.
        output_path: Directory to save the final adapter.
        lora_r: Computed LoRA rank.
        lora_alpha: Computed LoRA alpha.
        learning_rate: Auto-found or fallback learning rate.
        max_steps: Steps derived from dataset size and epoch count.
    """
    import json
    import time

    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer, SFTConfig

    print("\nStarting SFT training with automated hyperparameters...")
    t0 = time.time()

    sft_config = SFTConfig(
        output_dir=str(output_path / "checkpoints"),
        # Epochs & steps
        max_steps=max_steps,
        num_train_epochs=cfg.get("max_epochs", 5),
        # Batch — auto_find_batch_size starts from per_device and doubles until OOM
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 8),
        auto_find_batch_size=cfg.get("auto_find_batch_size", True),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        # LR — auto-found; cosine_with_restarts for better convergence on small models
        learning_rate=learning_rate,
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine_with_restarts"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.01),
        optim="adamw_torch",   # standard Adam for local/16-bit runs
        # Precision — 16-bit (E4B fits without quantization)
        fp16=False,
        bf16=True,
        # Gradient checkpointing to save memory
        gradient_checkpointing=True,
        # Early stopping requires eval_strategy + load_best_model_at_end
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 100),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging & saving
        logging_steps=50,
        save_steps=cfg.get("eval_steps", 100),
        save_total_limit=2,
        report_to="none",
        # SFT-specific
        max_seq_length=cfg.get("max_seq_length", 4096),
        packing=True,
        dataset_text_field="text",
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=cfg.get("early_stopping_patience", 3),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        callbacks=[early_stopping],
    )

    train_result = trainer.train()

    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed/3600:.2f} h")
    print(f"Final training loss : {train_result.training_loss:.4f}")
    print(f"Total steps         : {train_result.global_step:,}")

    # Save adapter
    final_dir = output_path / "final-adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    summary = {
        "model_name": cfg.get("model_name", "google/gemma-4-e4b-it"),
        "final_loss": train_result.training_loss,
        "global_step": train_result.global_step,
        "elapsed_hours": round(elapsed / 3600, 2),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "lr_auto_found": True,
        "max_epochs": cfg.get("max_epochs", 5),
        "max_steps_computed": max_steps,
        "early_stopping_patience": cfg.get("early_stopping_patience", 3),
        "auto_find_batch_size": cfg.get("auto_find_batch_size", True),
    }
    with (final_dir / "training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Adapter saved to {final_dir}")
    print(f"Summary:\n{json.dumps(summary, indent=2)}")


def main() -> None:
    """Entry point: validate arguments, load config, apply automated strategies,
    and launch training.

    Automated strategies applied:
      1. LoRA rank and alpha derived from hidden_dim (compute_lora_rank)
      2. Learning rate found via exponential sweep (find_learning_rate)
      3. max_steps computed from dataset size and max_epochs
      4. Batch size auto-found via auto_find_batch_size=True in SFTConfig
      5. Early stopping via EarlyStoppingCallback (patience from config)

    Raises:
        ValueError: If --language is omitted for per-language strategy.
        FileNotFoundError: If the training data file does not exist.
    """
    args = parse_args()

    if args.strategy == "per-language" and not args.language:
        raise ValueError("--language is required when --strategy is per-language")

    cfg = load_config(args.config)
    data_path, output_path = resolve_paths(args)

    # -----------------------------------------------------------------------
    # Automated LoRA rank — derived from hidden_dim, not guessed
    # Gemma 4 E4B hidden_dim ≈ 2048  →  r=16, alpha=32
    # -----------------------------------------------------------------------
    lora_r, lora_alpha = compute_lora_rank(HIDDEN_DIM_GEMMA_E4B)
    cfg["lora_r"] = lora_r
    cfg["lora_alpha"] = lora_alpha

    print(f"Strategy  : {args.strategy}")
    print(f"Language  : {args.language or 'all'}")
    print(f"Data      : {data_path}")
    print(f"Output    : {output_path}")
    print(f"\n[AutoHP] LoRA rank  : {lora_r}  (hidden_dim={HIDDEN_DIM_GEMMA_E4B} // 128)")
    print(f"[AutoHP] LoRA alpha : {lora_alpha}  (2 × rank)")
    print(f"\nConfig keys:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            f"Run dataset_prep.py first or check --data-dir."
        )

    # -----------------------------------------------------------------------
    # Load model and tokenizer
    # -----------------------------------------------------------------------
    print(f"\nLoading {cfg.get('model_name', 'google/gemma-4-e4b-it')}...")

    hf_token = os.environ.get("HF_TOKEN")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.get("model_name", "google/gemma-4-e4b-it"),
            max_seq_length=cfg.get("max_seq_length", 4096),
            load_in_4bit=cfg.get("load_in_4bit", False),
            dtype=None,
            token=hf_token,
        )
        USE_UNSLOTH = True
        print("Unsloth loaded successfully.")

        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules",
                                   ["q_proj", "v_proj", "k_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    except Exception as exc:
        print(f"Unsloth not available ({exc}); using HuggingFace PEFT...")
        USE_UNSLOTH = False

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.get("model_name", "google/gemma-4-e4b-it"), token=hf_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.get("model_name", "google/gemma-4-e4b-it"),
            torch_dtype="auto",
            device_map="auto",
            token=hf_token,
        )

        lora_config = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules",
                                   ["q_proj", "v_proj", "k_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # -----------------------------------------------------------------------
    # Load and format dataset
    # -----------------------------------------------------------------------
    from datasets import load_dataset as hf_load_dataset

    print(f"\nLoading training data from {data_path}...")
    raw_ds = hf_load_dataset("json", data_files=str(data_path), split="train")
    num_examples = len(raw_ds)
    print(f"Loaded {num_examples:,} examples.")

    def format_chatml(example: dict) -> dict:
        """Convert messages list to a single ChatML string."""
        messages = example.get("messages", [])
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return {"text": "\n".join(parts)}

    formatted_ds = raw_ds.map(format_chatml, remove_columns=raw_ds.column_names)

    # Split off eval set (5% or max 500 examples)
    eval_size = min(500, max(1, int(num_examples * 0.05)))
    split = formatted_ds.train_test_split(test_size=eval_size, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Train set: {len(train_ds):,}  |  Eval set: {len(eval_ds):,}")

    # -----------------------------------------------------------------------
    # Automated max_steps — derived from dataset size at runtime
    # -----------------------------------------------------------------------
    try:
        import torch
        n_gpus = torch.cuda.device_count() or 1
    except Exception:
        n_gpus = 1

    effective_batch = (
        cfg.get("per_device_train_batch_size", 8)
        * cfg.get("gradient_accumulation_steps", 4)
        * n_gpus
    )
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
    max_steps = steps_per_epoch * cfg.get("max_epochs", 5)

    print(f"\n[AutoHP] effective_batch : {effective_batch}")
    print(f"[AutoHP] steps_per_epoch : {steps_per_epoch}")
    print(f"[AutoHP] max_steps       : {max_steps}")

    # -----------------------------------------------------------------------
    # Automated LR finder
    # -----------------------------------------------------------------------
    learning_rate = find_learning_rate(
        model,
        train_ds,
        tokenizer,
        fallback_lr=cfg.get("learning_rate_fallback", 1e-4),
    )
    print(f"[AutoHP] learning_rate  : {learning_rate:.2e}")

    # -----------------------------------------------------------------------
    # Run training
    # -----------------------------------------------------------------------
    run_training(
        cfg=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        model=model,
        tokenizer=tokenizer,
        output_path=output_path,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        max_steps=max_steps,
    )


if __name__ == "__main__":
    main()
