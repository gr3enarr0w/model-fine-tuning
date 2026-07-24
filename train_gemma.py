"""
train_gemma.py — Gemma 4 E4B LoRA fine-tuning with automated hyperparameter strategies.

Two training strategies:
  Strategy A: per-language specialization (one adapter per language)
  Strategy B: generalist (single adapter, all languages)

Run locally on M2 Pro or on Modal A10G for faster iteration.

Usage (local):
  python train_gemma.py --strategy per-language --language python
  python train_gemma.py --strategy generalist

Usage (Modal — A10G at ~$1.10/hr):
  modal run train_gemma.py --language python
  modal run train_gemma.py --language python --max-steps 50   # smoke test
  modal run train_gemma.py                                    # generalist

Automated hyperparameter strategies
------------------------------------
  LoRA rank   : max(8, hidden_dim // 128)  — derived from model architecture
  LoRA alpha  : 2 × rank  — modern consensus, not rank==alpha
  Learning rate: exponential sweep (1e-7 → 1e-2 over 100 steps); finds steepest
                  descent point; falls back to config learning_rate_fallback
  Batch size  : auto_find_batch_size=True in TrainingArguments (TRL / HF Trainer)
  Epochs      : early stopping (patience=3) instead of a fixed epoch count
  max_steps   : derived at runtime from dataset size × epochs × batch
  Data volume : ACT controller — feeds 50k-example batches, stops when val loss
                improvement < epsilon (0.005); hard ceiling of 4 batches (200k)
"""
import argparse
import math
import os
from pathlib import Path

import modal
import yaml

# ---------------------------------------------------------------------------
# Modal app & shared resources
# ---------------------------------------------------------------------------

app = modal.App("gemma-e4b-codealchemy")

# Persistent volume — stores training data and output adapters.
# /data    → per-language JSONL files (train_python.jsonl, etc.) + train.jsonl
# /outputs → LoRA adapters saved after each language run
volume = modal.Volume.from_name("model-fine-tuning-vol", create_if_missing=True)

# Container image with all training dependencies.
# A10G supports BF16 natively; Unsloth with cu121-ampere wheels works well.
gemma_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "torchvision",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "peft>=0.13.0",
        "bitsandbytes>=0.44.0",
        "datasets>=3.0.0",
        "trl>=0.11.0",
        "sentencepiece",
        "protobuf",
        "pyyaml",
        "tqdm",
    )
    .pip_install(
        # Unsloth last — auto-detects torch/CUDA and compiles kernels
        "unsloth[cu121-ampere-torch240] @ https://github.com/unslothai/unsloth/archive/refs/heads/main.zip",
    )
)

# ---------------------------------------------------------------------------
# Modal training function — A10G (~$1.10/hr; sufficient for 4B param model)
# ---------------------------------------------------------------------------

@app.function(
    gpu="a10g",
    timeout=4 * 3600,          # 4-hour hard cap per language
    volumes={
        "/data": volume,
        "/outputs": volume,
    },
    image=gemma_image,
    secrets=[
        modal.Secret.from_name("huggingface-token"),
    ],
    memory=32768,              # 32 GB RAM; E4B is 4B params, fits comfortably
)
def train_modal(
    language: str | None = None,
    strategy: str = "per-language",
    max_steps_override: int = -1,
    use_act: bool = True,
    act_batch_size: int = 50_000,
    act_epsilon: float = 0.005,
    act_max_batches: int = 4,
) -> None:
    """
    QLoRA fine-tuning of Gemma 4 E4B on CodeAlchemy data — runs on Modal A10G.

    Args:
        language: Target language for per-language strategy (e.g. "python").
                  None triggers generalist training.
        strategy: "per-language" or "generalist".
        max_steps_override: Override computed max_steps for smoke testing.
        use_act: Enable ACT controller (default True). Set False for single-pass.
        act_batch_size: Examples per ACT data batch.
        act_epsilon: Minimum val loss improvement to continue to next batch.
        act_max_batches: Hard ceiling on number of ACT batches (budget guard).

    Saves adapter to /outputs/gemma-e4b-{language}/ or
    /outputs/gemma-e4b-generalist/ on the Modal volume.
    """
    import json
    import math
    import os
    import time
    from pathlib import Path as _Path

    import torch
    import yaml as _yaml

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN not set. Create the Modal secret:\n"
            "  modal secret create huggingface-token HF_TOKEN=hf_..."
        )

    # Resolve strategy from language argument
    if language is None:
        strategy = "generalist"
    else:
        strategy = "per-language"

    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------
    if strategy == "per-language":
        data_path = _Path(f"/data/train_{language}.jsonl")
        output_tag = f"gemma-e4b-{language}"
    else:
        data_path = _Path("/data/train.jsonl")
        output_tag = "gemma-e4b-generalist"

    output_path = _Path("/outputs") / output_tag
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Strategy  : {strategy}")
    print(f"Language  : {language or 'all (generalist)'}")
    print(f"Data      : {data_path}")
    print(f"Output    : {output_path}")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}.\n"
            "Upload it first:\n"
            f"  modal volume put model-fine-tuning-vol data/train_{language}.jsonl /data/train_{language}.jsonl"
        )

    # -----------------------------------------------------------------------
    # Config defaults (mirrors configs/lora_gemma.yaml)
    # -----------------------------------------------------------------------
    DEFAULTS: dict = {
        "model_name": "google/gemma-4-e4b-it",
        "max_seq_length": 4096,
        "load_in_4bit": False,    # E4B fits in BF16 on A10G without quantization
        "lora_r": 16,             # overridden below by hidden_dim formula
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
        "per_device_train_batch_size": 4,
        "auto_find_batch_size": True,
        "gradient_accumulation_steps": 4,
        "max_epochs": 5,
        "early_stopping_patience": 3,
        "eval_steps": 100,
        "load_best_model_at_end": True,
        "warmup_ratio": 0.03,
        "learning_rate_fallback": 1e-4,
        "lr_scheduler": "cosine_with_restarts",
        "weight_decay": 0.01,
        "save_total_limit": 2,
        "fp16": False,
        "bf16": True,
        "gradient_checkpointing": True,
        "packing": True,
    }
    cfg = DEFAULTS.copy()

    # -----------------------------------------------------------------------
    # Automated LoRA rank — Gemma 4 E4B hidden_dim=2048 → r=16, alpha=32
    # -----------------------------------------------------------------------
    lora_r = max(8, HIDDEN_DIM_GEMMA_E4B // 128)   # = 16
    lora_alpha = 2 * lora_r                         # = 32
    cfg["lora_r"] = lora_r
    cfg["lora_alpha"] = lora_alpha
    print(f"\n[AutoHP] LoRA rank  : {lora_r}  (hidden_dim={HIDDEN_DIM_GEMMA_E4B} // 128)")
    print(f"[AutoHP] LoRA alpha : {lora_alpha}  (2 × rank)")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    MODEL_NAME = cfg["model_name"]
    print(f"\nLoading {MODEL_NAME}...")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=cfg["max_seq_length"],
            load_in_4bit=cfg["load_in_4bit"],
            dtype=None,
            token=hf_token,
        )
        USE_UNSLOTH = True
        print("Unsloth loaded successfully.")

        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["lora_target_modules"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    except Exception as exc:
        print(f"Unsloth not available ({exc}); using HuggingFace PEFT...")
        USE_UNSLOTH = False

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
        )

        lora_config = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["lora_target_modules"],
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
    # ACT controller helpers (defined inside Modal function for lazy imports)
    # -----------------------------------------------------------------------

    def _load_jsonl(path: _Path) -> list[dict]:
        """Load a JSONL file into a list of dicts."""
        records = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _format_chatml(example: dict) -> dict:
        """Convert messages list to a single ChatML string."""
        messages = example.get("messages", [])
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return {"text": "\n".join(parts)}

    def _train_on_batch(
        model: object,
        tokenizer: object,
        batch_records: list[dict],
        val_path_str: str,
        cfg: dict,
        output_dir: str,
        max_steps_override: int = -1,
    ) -> float:
        """
        Train SFTTrainer on a single ACT batch; return best eval_loss.
        Returns float('inf') if training fails.
        """
        from datasets import Dataset as _HFDs
        from transformers import EarlyStoppingCallback
        from trl import SFTTrainer, SFTConfig

        batch_ds = _HFDs.from_list(batch_records)
        formatted_batch = batch_ds.map(_format_chatml, remove_columns=batch_ds.column_names)

        val_p = _Path(val_path_str)
        if val_p.exists():
            val_records = _load_jsonl(val_p)
            val_raw = _HFDs.from_list(val_records[:2000])
            eval_ds = val_raw.map(_format_chatml, remove_columns=val_raw.column_names)
        else:
            eval_size = min(500, max(1, int(len(formatted_batch) * 0.05)))
            split = formatted_batch.train_test_split(test_size=eval_size, seed=42)
            formatted_batch = split["train"]
            eval_ds = split["test"]

        train_ds = formatted_batch
        print(f"  [batch] Train: {len(train_ds):,}  |  Eval: {len(eval_ds):,}")

        n_gpus = torch.cuda.device_count() or 1
        effective_batch = (
            cfg["per_device_train_batch_size"]
            * cfg["gradient_accumulation_steps"]
            * n_gpus
        )
        steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
        batch_max_steps = steps_per_epoch * cfg["max_epochs"]

        if max_steps_override > 0:
            batch_max_steps = max_steps_override
            print(f"  [Smoke] --max-steps override: {batch_max_steps}")

        print(f"  [batch] effective_batch={effective_batch}  steps/epoch={steps_per_epoch}  max_steps={batch_max_steps}")

        checkpoints_dir = output_dir

        sft_config = SFTConfig(
            output_dir=checkpoints_dir,
            max_steps=batch_max_steps,
            num_train_epochs=cfg["max_epochs"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            auto_find_batch_size=cfg["auto_find_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg.get("_act_learning_rate", cfg.get("learning_rate_fallback", 1e-4)),
            lr_scheduler_type=cfg["lr_scheduler"],
            warmup_ratio=cfg["warmup_ratio"],
            weight_decay=cfg["weight_decay"],
            optim="adamw_torch",
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            eval_strategy="steps",
            eval_steps=cfg["eval_steps"],
            load_best_model_at_end=cfg["load_best_model_at_end"],
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=50,
            save_steps=cfg["eval_steps"],
            save_total_limit=cfg["save_total_limit"],
            report_to="none",
            max_seq_length=cfg["max_seq_length"],
            packing=True,
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=sft_config,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=cfg["early_stopping_patience"],
            )],
        )

        try:
            trainer.train()
            best_loss = getattr(trainer.state, "best_metric", None)
            if best_loss is None:
                eval_out = trainer.evaluate()
                best_loss = eval_out.get("eval_loss", float("inf"))
            return float(best_loss)
        except Exception as exc:
            print(f"  [batch] Training failed: {exc}")
            return float("inf")

    def _act_controller(
        model: object,
        tokenizer: object,
        cfg: dict,
        data_path_str: str,
        val_path_str: str,
        output_dir: str,
        batch_size: int = 50_000,
        epsilon: float = 0.005,
        max_batches: int = 4,
        max_steps_override: int = -1,
    ) -> float:
        """
        ACT (Auto-Train Controller): feeds data in batches, stops when val loss
        improvement drops below epsilon. Returns final best val loss.

        Based on: ACT: Auto-Train for Code Translation Framework (2025)
        """
        print(f"\n[ACT] Loading full dataset from {data_path_str}...")
        all_records = _load_jsonl(_Path(data_path_str))
        total = len(all_records)
        print(f"[ACT] Total records: {total:,}  |  batch_size={batch_size:,}  |  max_batches={max_batches}  |  ε={epsilon}")

        best_val_loss = float("inf")

        for batch_idx in range(max_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total)
            batch = all_records[batch_start:batch_end]

            if not batch:
                print(f"[ACT] No more data at batch {batch_idx}. Stopping.")
                break

            print(f"\n[ACT] Batch {batch_idx + 1}/{max_batches}: {len(batch):,} examples "
                  f"(rows {batch_start:,}–{batch_end:,}; total seen: {batch_end:,})")

            batch_val_loss = _train_on_batch(
                model,
                tokenizer,
                batch,
                val_path_str,
                cfg,
                output_dir,
                max_steps_override=max_steps_override,
            )

            improvement = best_val_loss - batch_val_loss
            print(f"[ACT] Val loss: {batch_val_loss:.4f} | Best: {best_val_loss:.4f} "
                  f"| Improvement: {improvement:.4f} | ε={epsilon}")

            if batch_val_loss < best_val_loss:
                best_val_loss = batch_val_loss

            if batch_idx > 0 and improvement < epsilon:
                print(f"[ACT] Improvement {improvement:.4f} < ε {epsilon}. "
                      f"Data saturated. Stopping.")
                break

            if batch_end >= total:
                print(f"[ACT] All {total:,} examples consumed. Stopping.")
                break

            print(f"[ACT] Still improving. Loading next batch...")

        print(f"\n[ACT] Final best val loss: {best_val_loss:.4f}")
        return best_val_loss

    # -----------------------------------------------------------------------
    # Automated LR finder (sweep on small stub, store result in cfg)
    # -----------------------------------------------------------------------
    def _find_lr(model, train_stub_ds, tokenizer, fallback_lr=1e-4):
        """Exponential LR sweep; returns optimal LR or fallback on failure."""
        import copy

        print(f"\n[LRFinder] Sweeping LR 1e-7 → 1e-2 over 100 steps...")
        try:
            from torch.optim import AdamW

            sweep_model = copy.deepcopy(model)
            sweep_model.train()
            optimizer = AdamW(sweep_model.parameters(), lr=1e-7)

            try:
                device = next(sweep_model.parameters()).device
            except StopIteration:
                device = "cuda"

            sweep_texts = [train_stub_ds[i]["text"] for i in range(min(200, len(train_stub_ds)))]
            encodings = tokenizer(
                sweep_texts, truncation=True, max_length=256,
                padding="max_length", return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            lr_multiplier = (1e-2 / 1e-7) ** (1.0 / 100)
            losses: list[float] = []
            lrs: list[float] = []
            current_lr = 1e-7

            for step in range(100):
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr
                idx = step % len(input_ids)
                optimizer.zero_grad()
                outputs = sweep_model(
                    input_ids=input_ids[idx:idx+1],
                    attention_mask=attention_mask[idx:idx+1],
                    labels=input_ids[idx:idx+1],
                )
                outputs.loss.backward()
                optimizer.step()
                losses.append(outputs.loss.item())
                lrs.append(current_lr)
                current_lr *= lr_multiplier
                if outputs.loss.item() > 10 * losses[0]:
                    break

            beta, avg, smoothed = 0.9, losses[0], []
            for l in losses:
                avg = beta * avg + (1 - beta) * l
                smoothed.append(avg / (1 - beta ** (len(smoothed) + 1)))

            if len(smoothed) < 3:
                raise ValueError("Too few steps")

            gradients = [smoothed[i+1] - smoothed[i] for i in range(len(smoothed) - 1)]
            best_idx = gradients.index(min(gradients))
            suggested_lr = float(min(max(lrs[max(0, best_idx - 1)], 1e-6), 5e-4))

            del sweep_model
            torch.cuda.empty_cache()
            print(f"[LRFinder] Suggested LR: {suggested_lr:.2e}")
            return suggested_lr

        except Exception as exc:
            print(f"[LRFinder] Sweep failed ({exc}); using fallback {fallback_lr:.0e}")
            return fallback_lr

    # Build a small stub dataset for LR sweep from the first 200 records
    from datasets import Dataset as _HFDsLR
    _lr_stub_records = _load_jsonl(data_path)[:200]
    _lr_stub_raw = _HFDsLR.from_list(_lr_stub_records)
    _lr_stub_ds = _lr_stub_raw.map(_format_chatml, remove_columns=_lr_stub_raw.column_names)

    learning_rate = _find_lr(model, _lr_stub_ds, tokenizer,
                              fallback_lr=cfg["learning_rate_fallback"])
    print(f"[AutoHP] learning_rate  : {learning_rate:.2e}")

    # Store in cfg so _train_on_batch (called inside _act_controller) can use it
    cfg["_act_learning_rate"] = learning_rate

    # -----------------------------------------------------------------------
    # Run ACT controller (or single-pass if use_act=False)
    # -----------------------------------------------------------------------
    VAL_PATH = str(data_path.parent / "val.jsonl")
    checkpoints_dir = str(output_path / "checkpoints")

    print("\nStarting ACT-controlled training...")
    t0 = time.time()

    if use_act:
        print(f"[ACT] mode=ON  batch_size={act_batch_size:,}  epsilon={act_epsilon}  max_batches={act_max_batches}")
        final_val_loss = _act_controller(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            data_path_str=str(data_path),
            val_path_str=VAL_PATH,
            output_dir=checkpoints_dir,
            batch_size=act_batch_size,
            epsilon=act_epsilon,
            max_batches=act_max_batches,
            max_steps_override=max_steps_override,
        )
    else:
        print("[ACT] mode=OFF — single-pass training on full dataset")
        all_records = _load_jsonl(data_path)
        final_val_loss = _train_on_batch(
            model=model,
            tokenizer=tokenizer,
            batch_records=all_records,
            val_path_str=VAL_PATH,
            cfg=cfg,
            output_dir=checkpoints_dir,
            max_steps_override=max_steps_override,
        )

    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed/3600:.2f} h")
    print(f"Final best val loss : {final_val_loss:.4f}")

    # -----------------------------------------------------------------------
    # Save adapter
    # -----------------------------------------------------------------------
    final_dir = output_path / "final-adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    summary = {
        "model_name": MODEL_NAME,
        "strategy": strategy,
        "language": language,
        "output_tag": output_tag,
        "final_val_loss": final_val_loss,
        "elapsed_hours": round(elapsed / 3600, 2),
        "gpu": "a10g",
        "use_unsloth": USE_UNSLOTH,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        # ACT controller parameters
        "act_enabled": use_act,
        "act_batch_size": act_batch_size,
        "act_epsilon": act_epsilon,
        "act_max_batches": act_max_batches,
    }
    with (final_dir / "training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Adapter saved to {final_dir}")
    print(f"Summary:\n{json.dumps(summary, indent=2)}")

    # Commit volume so adapter persists after the container exits
    volume.commit()
    print(f"\nVolume committed. Adapter at /outputs/{output_tag}/final-adapter/")

    print("\n=== Training complete ===")


# ---------------------------------------------------------------------------
# Modal local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def modal_main(
    language: str = "",
    max_steps: int = -1,
    no_act: bool = False,
    act_batch_size: int = 50_000,
    act_epsilon: float = 0.005,
    act_max_batches: int = 4,
) -> None:
    """
    Trigger the remote Gemma E4B training job on Modal A10G.

    Run with:
        modal run train_gemma.py --language python
        modal run train_gemma.py --language python --max-steps 50   # smoke test
        modal run train_gemma.py                                    # generalist
        modal run train_gemma.py --no-act                           # disable ACT controller

    Data volume is governed by the ACT controller (epsilon=0.005, max 4 batches = 200k examples).
    No --limit flag needed; the ACT batch ceiling replaces it.

    Retrieve adapter after training:
        modal volume get model-fine-tuning-vol /outputs/gemma-e4b-python/final-adapter ./gemma-python-adapter
    """
    lang = language.strip() or None
    strategy = "per-language" if lang else "generalist"
    tag = f"gemma-e4b-{lang}" if lang else "gemma-e4b-generalist"

    print("Submitting Gemma E4B training job to Modal...")
    print(f"  GPU       : A10G (~$1.10/hr)")
    print(f"  Timeout   : 4 hours")
    print(f"  Strategy  : {strategy}")
    print(f"  Language  : {lang or 'all (generalist)'}")
    print(f"  Volume    : model-fine-tuning-vol")
    print(f"  Output    : /outputs/{tag}/")
    print(f"  ACT       : {'OFF (single-pass)' if no_act else 'ON'}")
    if not no_act:
        print(f"  ACT batch size  : {act_batch_size:,}")
        print(f"  ACT epsilon     : {act_epsilon}")
        print(f"  ACT max batches : {act_max_batches}  (max {act_batch_size * act_max_batches:,} examples)")
    if max_steps > 0:
        print(f"  [Smoke] max-steps : {max_steps}")
    print()

    train_modal.remote(
        language=lang,
        strategy=strategy,
        max_steps_override=max_steps,
        use_act=not no_act,
        act_batch_size=act_batch_size,
        act_epsilon=act_epsilon,
        act_max_batches=act_max_batches,
    )

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
    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override max training steps, -1 = auto from dataset size (e.g. --max-steps 50 for smoke test)",
    )
    p.add_argument(
        "--no-act",
        action="store_true",
        default=False,
        help="Disable ACT controller; train single-pass on full dataset.",
    )
    p.add_argument(
        "--act-batch-size",
        type=int,
        default=50_000,
        help="Number of examples per ACT data batch (default: 50000).",
    )
    p.add_argument(
        "--act-epsilon",
        type=float,
        default=0.005,
        help="Minimum val loss improvement to continue to next ACT batch (default: 0.005).",
    )
    p.add_argument(
        "--act-max-batches",
        type=int,
        default=4,
        help="Hard ceiling on ACT batches — max examples = batch_size × max_batches (default: 4).",
    )
    args = p.parse_args()
    if args.strategy == "per-language" and not args.language:
        p.error("--language is required when --strategy is per-language")
    if args.strategy == "generalist" and args.language:
        p.error("--language must not be set when --strategy is generalist")
    return args


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
        FileNotFoundError: If the training data file does not exist.
    """
    args = parse_args()

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
            target_modules=cfg.get("target_modules",
                                   cfg.get("lora_target_modules",
                                           ["q_proj", "v_proj", "k_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj"])),
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
            target_modules=cfg.get("target_modules",
                                   cfg.get("lora_target_modules",
                                           ["q_proj", "v_proj", "k_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj"])),
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
    # Automated LR finder — run on small stub (200 records) for speed
    # -----------------------------------------------------------------------
    import json as _json_main

    def _load_jsonl_main(path: Path) -> list[dict]:
        records = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(_json_main.loads(line))
                    except _json_main.JSONDecodeError:
                        pass
        return records

    def _fmt_chatml_main(example: dict) -> dict:
        messages = example.get("messages", [])
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return {"text": "\n".join(parts)}

    from datasets import Dataset as _HFDsMain

    _lr_stub_recs = _load_jsonl_main(data_path)[:200]
    _lr_stub_raw = _HFDsMain.from_list(_lr_stub_recs)
    _lr_stub_fmt = _lr_stub_raw.map(_fmt_chatml_main, remove_columns=_lr_stub_raw.column_names)

    learning_rate = find_learning_rate(
        model,
        _lr_stub_fmt,
        tokenizer,
        fallback_lr=cfg.get("learning_rate_fallback", 1e-4),
    )
    print(f"[AutoHP] learning_rate  : {learning_rate:.2e}")

    # Store found LR in cfg for use inside act_controller_main
    cfg["_act_learning_rate"] = learning_rate

    # -----------------------------------------------------------------------
    # ACT controller (local version — mirrors the Modal function's implementation)
    # -----------------------------------------------------------------------
    use_act: bool = not args.no_act
    act_batch_size: int = args.act_batch_size
    act_epsilon: float = args.act_epsilon
    act_max_batches: int = args.act_max_batches

    val_path = data_path.parent / "val.jsonl"
    checkpoints_dir = output_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _train_on_batch_main(
        batch_records: list[dict],
        val_path_str: str,
        cfg: dict,
        output_dir: str,
    ) -> float:
        """Thin wrapper — runs SFTTrainer on a batch slice and returns best eval_loss."""
        from datasets import Dataset as _HFDsBatch

        batch_ds = _HFDsBatch.from_list(batch_records)
        formatted_batch = batch_ds.map(_fmt_chatml_main, remove_columns=batch_ds.column_names)

        val_p = Path(val_path_str)
        if val_p.exists():
            val_recs = _load_jsonl_main(val_p)[:2000]
            val_raw = _HFDsBatch.from_list(val_recs)
            eval_ds = val_raw.map(_fmt_chatml_main, remove_columns=val_raw.column_names)
        else:
            eval_size = min(500, max(1, int(len(formatted_batch) * 0.05)))
            split = formatted_batch.train_test_split(test_size=eval_size, seed=42)
            formatted_batch = split["train"]
            eval_ds = split["test"]

        train_ds = formatted_batch

        try:
            import torch as _torch
            n_gpus = _torch.cuda.device_count() or 1
        except Exception:
            n_gpus = 1

        effective_batch = (
            cfg.get("per_device_train_batch_size", 8)
            * cfg.get("gradient_accumulation_steps", 4)
            * n_gpus
        )
        steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
        batch_max_steps = steps_per_epoch * cfg.get("max_epochs", 5)
        if args.max_steps > 0:
            batch_max_steps = args.max_steps

        print(f"  [batch] Train={len(train_ds):,}  Eval={len(eval_ds):,}  max_steps={batch_max_steps}")

        from transformers import EarlyStoppingCallback
        from trl import SFTTrainer, SFTConfig

        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=batch_max_steps,
            num_train_epochs=cfg.get("max_epochs", 5),
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 8),
            auto_find_batch_size=cfg.get("auto_find_batch_size", True),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
            learning_rate=cfg.get("_act_learning_rate", cfg.get("learning_rate_fallback", 1e-4)),
            lr_scheduler_type=cfg.get("lr_scheduler", "cosine_with_restarts"),
            warmup_ratio=cfg.get("warmup_ratio", 0.03),
            weight_decay=cfg.get("weight_decay", 0.01),
            optim="adamw_torch",
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            eval_strategy="steps",
            eval_steps=cfg.get("eval_steps", 100),
            load_best_model_at_end=cfg.get("load_best_model_at_end", True),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=50,
            save_steps=cfg.get("eval_steps", 100),
            save_total_limit=2,
            report_to="none",
            max_seq_length=cfg.get("max_seq_length", 4096),
            packing=True,
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=sft_config,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=cfg.get("early_stopping_patience", 3),
            )],
        )
        try:
            trainer.train()
            best_loss = getattr(trainer.state, "best_metric", None)
            if best_loss is None:
                eval_out = trainer.evaluate()
                best_loss = eval_out.get("eval_loss", float("inf"))
            return float(best_loss)
        except Exception as exc:
            print(f"  [batch] Training failed: {exc}")
            return float("inf")

    print(f"\n[ACT] mode={'ON' if use_act else 'OFF'}  batch_size={act_batch_size:,}  epsilon={act_epsilon}  max_batches={act_max_batches}")

    if use_act:
        all_records = _load_jsonl_main(data_path)
        total = len(all_records)
        print(f"[ACT] Total records: {total:,}")
        best_val_loss = float("inf")

        for batch_idx in range(act_max_batches):
            batch_start = batch_idx * act_batch_size
            batch_end = min(batch_start + act_batch_size, total)
            batch = all_records[batch_start:batch_end]

            if not batch:
                print(f"[ACT] No more data at batch {batch_idx}. Stopping.")
                break

            print(f"\n[ACT] Batch {batch_idx + 1}/{act_max_batches}: {len(batch):,} examples "
                  f"(rows {batch_start:,}–{batch_end:,}; total seen: {batch_end:,})")

            batch_val_loss = _train_on_batch_main(
                batch_records=batch,
                val_path_str=str(val_path),
                cfg=cfg,
                output_dir=str(checkpoints_dir),
            )

            improvement = best_val_loss - batch_val_loss
            print(f"[ACT] Val loss: {batch_val_loss:.4f} | Best: {best_val_loss:.4f} "
                  f"| Improvement: {improvement:.4f} | ε={act_epsilon}")

            if batch_val_loss < best_val_loss:
                best_val_loss = batch_val_loss

            if batch_idx > 0 and improvement < act_epsilon:
                print(f"[ACT] Improvement {improvement:.4f} < ε {act_epsilon}. "
                      f"Data saturated. Stopping.")
                break

            if batch_end >= total:
                print(f"[ACT] All {total:,} examples consumed. Stopping.")
                break

            print(f"[ACT] Still improving. Loading next batch...")

        print(f"\n[ACT] Final best val loss: {best_val_loss:.4f}")
        final_val_loss = best_val_loss
    else:
        print("[ACT] Single-pass mode — training on full dataset")
        all_records = _load_jsonl_main(data_path)
        final_val_loss = _train_on_batch_main(
            batch_records=all_records,
            val_path_str=str(val_path),
            cfg=cfg,
            output_dir=str(checkpoints_dir),
        )

    # -----------------------------------------------------------------------
    # Save adapter (mirrors run_training output structure)
    # -----------------------------------------------------------------------
    final_dir = output_path / "final-adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    import json as _json_save
    summary = {
        "model_name": cfg.get("model_name", "google/gemma-4-e4b-it"),
        "strategy": args.strategy,
        "language": args.language,
        "final_val_loss": final_val_loss,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "lr_auto_found": True,
        "max_epochs": cfg.get("max_epochs", 5),
        "early_stopping_patience": cfg.get("early_stopping_patience", 3),
        "auto_find_batch_size": cfg.get("auto_find_batch_size", True),
        "act_enabled": use_act,
        "act_batch_size": act_batch_size,
        "act_epsilon": act_epsilon,
        "act_max_batches": act_max_batches,
    }
    with (final_dir / "training_summary.json").open("w") as f:
        _json_save.dump(summary, f, indent=2)

    print(f"Adapter saved to {final_dir}")
    print(f"Summary:\n{_json_save.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
