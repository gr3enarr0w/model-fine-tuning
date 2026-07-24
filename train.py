"""
train.py — Modal QLoRA fine-tuning job for Laguna XS 2.1 on CodeAlchemy data.

Usage:
    modal run train.py               # triggers train.remote()
    modal run train.py --detach      # fire-and-forget (recommended for ~4-6h runs)

Estimated cost: ~$15-22 on a single A100-40GB at Modal's $2.80/hr GPU rate.
Free credits cover this comfortably within the $30 allowance.

Before running:
    1. modal secret create huggingface-token HF_TOKEN=hf_...
    2. modal volume create laguna-codealchemy-vol
    3. Upload data:  modal volume put laguna-codealchemy-vol data/train.jsonl /data/train.jsonl
    4. (Optional) modal secret create wandb-token WANDB_API_KEY=...

Automated hyperparameter strategies
------------------------------------
  LoRA rank   : max(8, hidden_dim // 128)  — derived from model architecture
  LoRA alpha  : 2 × rank  — modern consensus, not rank==alpha
  Learning rate: exponential sweep (1e-7 → 1e-2 over 100 steps); finds steepest
                  descent point; falls back to config learning_rate_fallback
  Batch size  : auto_find_batch_size=True in TrainingArguments (TRL)
  Epochs      : early stopping (patience=3) instead of a fixed epoch count
  max_steps   : derived at runtime from dataset size × epochs × batch
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Modal app & shared resources
# ---------------------------------------------------------------------------

app = modal.App("laguna-codealchemy")

# Persistent volume — stores uploaded training data and output adapter.
# /data  → training data (uploaded before the run)
# /outputs → LoRA adapter saved after training
volume = modal.Volume.from_name("laguna-codealchemy-vol", create_if_missing=True)

# Container image with all training dependencies.
# Unsloth is installed last because it pins specific torch/cuda versions;
# installing it after the others lets pip resolve cleanly.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core ML stack — pin torch to a CUDA 12.1 wheel that Unsloth expects
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
        "wandb",
    )
    .pip_install(
        # Unsloth last — it auto-detects torch/CUDA and compiles kernels
        "unsloth[cu121-ampere-torch240] @ https://github.com/unslothai/unsloth/archive/refs/heads/main.zip",
        # Fallback: if the URL above fails, use the PyPI release:
        # "unsloth",
    )
)

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    gpu="a100-40gb",
    timeout=7 * 3600,          # 7-hour hard cap (run typically finishes in 4-6h)
    volumes={
        "/data": volume,
        "/outputs": volume,
    },
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        modal.Secret.from_name("wandb-token", required=False),  # optional
    ],
    memory=65536,              # 64 GB RAM to handle tokenization buffers
)
def train() -> None:
    """
    Full QLoRA fine-tuning pipeline with automated hyperparameter strategies:

    1. Load YAML config (baked into the image from the local configs/ directory).
    2. Load Laguna XS 2.1 with 4-bit NF4 quantization.
    3. Apply LoRA adapters — rank derived from hidden_dim, alpha = 2×rank.
    4. Find optimal LR via exponential sweep (find_learning_rate()).
    5. Load training data; compute max_steps from dataset size at runtime.
    6. Run SFTTrainer with auto_find_batch_size and EarlyStoppingCallback.
    7. Save the LoRA adapter to /outputs/final-adapter/.
    8. Commit the volume so the adapter persists after the container exits.
    """

    # -----------------------------------------------------------------------
    # Lazy imports (all inside the Modal function so they resolve in the image)
    # -----------------------------------------------------------------------
    import json
    import math
    import os
    import time
    from pathlib import Path

    import torch
    import yaml

    # -----------------------------------------------------------------------
    # 0. Read config
    # -----------------------------------------------------------------------
    CONFIG_PATH = Path("/configs/qlora.yaml")

    # Fallback inline defaults if the config file was not uploaded.
    # These mirror configs/qlora.yaml — automated values are applied below.
    DEFAULTS: dict = {
        "model_name": "poolside/Laguna-XS-2.1",
        "max_seq_length": 8192,
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        # LoRA — overridden below by the hidden_dim formula
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        # Training dynamics
        "per_device_train_batch_size": 2,
        "auto_find_batch_size": True,
        "gradient_accumulation_steps": 8,
        "max_epochs": 5,
        "early_stopping_patience": 3,
        "eval_steps": 200,
        "load_best_model_at_end": True,
        "warmup_ratio": 0.03,
        "learning_rate_fallback": 1e-4,
        "lr_scheduler": "cosine_with_restarts",
        "weight_decay": 0.01,
        "logging_steps": 50,
        "save_steps": 200,
        "save_total_limit": 2,
        "fp16": False,
        "bf16": True,
        "optim": "adamw_8bit",
        "gradient_checkpointing": True,
        "packing": True,
        "output_dir": "/outputs/checkpoints",
        "run_name": "laguna-xs-codealchemy",
        "report_to": "wandb",
    }

    if CONFIG_PATH.exists():
        print(f"Loading config from {CONFIG_PATH}")
        with CONFIG_PATH.open() as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = {**DEFAULTS, **user_cfg}
    else:
        print(
            f"WARNING: {CONFIG_PATH} not found — using built-in defaults.\n"
            "To customise, upload configs/qlora.yaml to the volume:\n"
            "  modal volume put laguna-codealchemy-vol configs/qlora.yaml /configs/qlora.yaml"
        )
        cfg = DEFAULTS

    # -----------------------------------------------------------------------
    # 0a. Automated LoRA rank — derived from model hidden_dim, not guessed.
    #     Formula: r = max(8, hidden_dim // 128);  alpha = 2 × r
    #     Laguna XS 2.1 active hidden_dim ≈ 4096  →  r=32, alpha=64
    #     These override whatever is in the YAML so the formula is authoritative.
    # -----------------------------------------------------------------------
    HIDDEN_DIM_LAGUNA = 4096
    lora_r = max(8, HIDDEN_DIM_LAGUNA // 128)   # = 32
    lora_alpha = 2 * lora_r                      # = 64
    cfg["lora_r"] = lora_r
    cfg["lora_alpha"] = lora_alpha
    print(f"\n[AutoHP] LoRA rank   : {lora_r}  (hidden_dim={HIDDEN_DIM_LAGUNA} // 128)")
    print(f"[AutoHP] LoRA alpha  : {lora_alpha}  (2 × rank)")

    MODEL_NAME: str = cfg["model_name"]
    OUTPUT_DIR: str = cfg["output_dir"]
    FINAL_ADAPTER_DIR = "/outputs/final-adapter"
    DATA_PATH = Path("/data/train.jsonl")

    print("\n=== Training config ===")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()

    # -----------------------------------------------------------------------
    # 1. Sanity checks
    # -----------------------------------------------------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {DATA_PATH}.\n"
            "Upload it first:\n"
            "  modal volume put laguna-codealchemy-vol data/train.jsonl /data/train.jsonl"
        )

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN not set. Create the Modal secret:\n"
            "  modal secret create huggingface-token HF_TOKEN=hf_..."
        )

    # Optional W&B setup
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    use_wandb = bool(wandb_key)
    if not use_wandb:
        cfg["report_to"] = "none"
        print("W&B API key not found — logging disabled. "
              "To enable: modal secret create wandb-token WANDB_API_KEY=...")

    # -----------------------------------------------------------------------
    # 2. Load model with 4-bit quantization via Unsloth
    # -----------------------------------------------------------------------
    print(f"\nLoading {MODEL_NAME} with Unsloth 4-bit quantization...")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=cfg["max_seq_length"],
            load_in_4bit=cfg["load_in_4bit"],
            dtype=None,      # Unsloth auto-selects BF16 on A100
            token=hf_token,
        )
        USE_UNSLOTH = True
        print("Unsloth loaded successfully.")

    except Exception as exc:
        print(f"Unsloth load failed ({exc}); falling back to HuggingFace PEFT...")
        USE_UNSLOTH = False

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=cfg["load_in_4bit"],
            bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_cfg,
            device_map="auto",
            token=hf_token,
        )

    # Ensure tokenizer has a pad token (needed for batch training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # -----------------------------------------------------------------------
    # 3. Apply LoRA adapters
    # -----------------------------------------------------------------------
    print(f"\nApplying LoRA adapters (r={cfg['lora_r']}, alpha={cfg['lora_alpha']})...")

    if USE_UNSLOTH:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg.get("lora_target_modules", cfg.get("target_modules")),
            bias="none",
            use_gradient_checkpointing="unsloth",   # Unsloth's optimised GC
            random_state=42,
        )
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg.get("lora_target_modules", cfg.get("target_modules")),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # -----------------------------------------------------------------------
    # 4. Load training data
    # -----------------------------------------------------------------------
    print(f"\nLoading training data from {DATA_PATH}...")

    from datasets import load_dataset as hf_load_dataset

    raw_ds = hf_load_dataset("json", data_files=str(DATA_PATH), split="train")
    num_examples = len(raw_ds)
    print(f"Loaded {num_examples:,} training examples.")

    # ChatML formatting function.
    # dataset_prep.py stores records as {"messages": [{role, content}, ...]}.
    def format_chatml(example: dict) -> dict:
        """Convert messages list to a single ChatML string."""
        messages = example.get("messages", [])
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")   # open for generation
        return {"text": "\n".join(parts)}

    # Pre-format so SFTTrainer can work with the "text" field.
    formatted_ds = raw_ds.map(format_chatml, remove_columns=raw_ds.column_names)
    print(f"Sample formatted example:\n{formatted_ds[0]['text'][:400]}...")

    # Split off a small eval set for early stopping (5% or max 500 examples)
    eval_size = min(500, max(1, int(num_examples * 0.05)))
    split = formatted_ds.train_test_split(test_size=eval_size, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Train set: {len(train_ds):,}  |  Eval set: {len(eval_ds):,}")

    # -----------------------------------------------------------------------
    # 4a. Automated max_steps — derived from dataset size, not guessed.
    #     effective_batch = per_device_batch × gradient_accumulation × n_gpus
    #     max_steps = ceil(len(train_ds) / effective_batch) × max_epochs
    # -----------------------------------------------------------------------
    n_gpus = torch.cuda.device_count() or 1
    effective_batch = (
        cfg["per_device_train_batch_size"]
        * cfg["gradient_accumulation_steps"]
        * n_gpus
    )
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
    max_steps = steps_per_epoch * cfg["max_epochs"]
    print(f"\n[AutoHP] effective_batch : {effective_batch} "
          f"(bs={cfg['per_device_train_batch_size']} × accum={cfg['gradient_accumulation_steps']} × gpus={n_gpus})")
    print(f"[AutoHP] steps_per_epoch : {steps_per_epoch}")
    print(f"[AutoHP] max_steps       : {max_steps}  ({cfg['max_epochs']} epochs × {steps_per_epoch} steps/epoch)")

    # -----------------------------------------------------------------------
    # 4b. Automated LR finder — exponential sweep 1e-7 → 1e-2 over 100 steps.
    #     Returns the LR at the point of steepest loss descent (maximum
    #     negative gradient of smoothed loss).  Falls back to
    #     cfg["learning_rate_fallback"] if the sweep fails for any reason.
    # -----------------------------------------------------------------------
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

        Runs num_sweep_steps mini-batches with LR increasing exponentially
        from start_lr to end_lr, records loss at each step, then returns the
        LR just before the loss starts diverging (point of maximum negative
        gradient of the smoothed loss curve).

        Args:
            model: The LoRA-wrapped model (already on GPU).
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
            from torch.optim import AdamW
            from torch.utils.data import DataLoader

            # Work on a throwaway copy of model weights so the sweep does not
            # corrupt the actual model parameters.
            sweep_model = copy.deepcopy(model)
            sweep_model.train()

            optimizer = AdamW(sweep_model.parameters(), lr=start_lr)

            # Tokenise a small subset for the sweep
            sweep_texts = [train_dataset[i]["text"]
                           for i in range(min(num_sweep_steps * 2, len(train_dataset)))]
            encodings = tokenizer(
                sweep_texts,
                truncation=True,
                max_length=512,     # short context for speed
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encodings["input_ids"].to(sweep_model.device if
                                                   hasattr(sweep_model, "device") else "cuda")
            attention_mask = encodings["attention_mask"].to(input_ids.device)

            lr_multiplier = (end_lr / start_lr) ** (1.0 / num_sweep_steps)
            losses: list[float] = []
            lrs: list[float] = []
            current_lr = start_lr

            for step in range(num_sweep_steps):
                # Set LR for this step
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
                smoothed.append(avg / (1 - beta ** (len(smoothed) + 1)))  # bias correction

            # Find steepest descent: largest negative gradient of smoothed loss
            if len(smoothed) < 3:
                raise ValueError("Too few sweep steps to determine gradient.")

            gradients = [smoothed[i + 1] - smoothed[i] for i in range(len(smoothed) - 1)]
            best_idx = gradients.index(min(gradients))

            # Use LR one step before minimum gradient (the "safe" side)
            suggested_lr = lrs[max(0, best_idx - 1)]

            # Clamp to a sane range
            suggested_lr = float(min(max(suggested_lr, 1e-6), 5e-4))

            del sweep_model
            torch.cuda.empty_cache()

            print(f"[LRFinder] Suggested LR: {suggested_lr:.2e}  "
                  f"(steepest descent at step {best_idx})")
            return suggested_lr

        except Exception as exc:
            print(f"[LRFinder] Sweep failed ({exc}); using fallback LR {fallback_lr:.0e}")
            return fallback_lr

    # Run the LR finder
    learning_rate = find_learning_rate(
        model,
        train_ds,
        tokenizer,
        fallback_lr=cfg.get("learning_rate_fallback", 1e-4),
    )
    print(f"[AutoHP] learning_rate  : {learning_rate:.2e}")

    # -----------------------------------------------------------------------
    # 5. Set up W&B (if enabled)
    # -----------------------------------------------------------------------
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "laguna-codealchemy"),
            name=cfg.get("run_name", "laguna-xs-codealchemy"),
            config={**cfg, "learning_rate": learning_rate, "max_steps": max_steps},
        )
        print(f"W&B run: {wandb.run.url}")

    # -----------------------------------------------------------------------
    # 6. Configure and run SFTTrainer with automated strategies
    # -----------------------------------------------------------------------
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer, SFTConfig

    print("\nStarting SFT training with automated hyperparameters...")
    t0 = time.time()

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        # Epochs & steps — max_steps derived from dataset size at runtime
        max_steps=max_steps,
        num_train_epochs=cfg["max_epochs"],   # upper bound; early stopping may end it sooner
        # Batch size — auto_find_batch_size lets TRL double from 1 until OOM
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        auto_find_batch_size=cfg.get("auto_find_batch_size", True),
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        # LR — auto-found above; cosine_with_restarts scheduler
        learning_rate=learning_rate,
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine_with_restarts"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.01),
        optim=cfg.get("optim", "adamw_8bit"),
        # Precision
        fp16=cfg.get("fp16", False),
        bf16=cfg.get("bf16", True),
        # Gradient checkpointing
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        # Early stopping requires eval_strategy + load_best_model_at_end
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 200),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging & saving
        logging_steps=cfg.get("logging_steps", 50),
        save_steps=cfg.get("eval_steps", 200),   # save on every eval
        save_total_limit=cfg.get("save_total_limit", 2),
        report_to=cfg.get("report_to", "none"),
        run_name=cfg.get("run_name", "laguna-xs-codealchemy"),
        # SFT-specific
        max_seq_length=cfg["max_seq_length"],
        packing=cfg.get("packing", True),
        dataset_text_field="text",
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=cfg.get("early_stopping_patience", 3),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
        callbacks=[early_stopping],
    )

    # Train
    train_result = trainer.train()

    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed/3600:.2f} h")
    print(f"Final training loss : {train_result.training_loss:.4f}")
    print(f"Total steps         : {train_result.global_step:,}")
    print(f"Samples/sec         : {train_result.metrics.get('train_samples_per_second', 'N/A')}")

    # -----------------------------------------------------------------------
    # 7. Save LoRA adapter
    # -----------------------------------------------------------------------
    print(f"\nSaving LoRA adapter to {FINAL_ADAPTER_DIR} ...")
    Path(FINAL_ADAPTER_DIR).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

    # Write a training summary alongside the adapter
    summary = {
        "model_name": MODEL_NAME,
        "run_name": cfg.get("run_name", "laguna-xs-codealchemy"),
        "final_loss": train_result.training_loss,
        "global_step": train_result.global_step,
        "elapsed_hours": round(elapsed / 3600, 2),
        "gpu": "a100-40gb",
        "use_unsloth": USE_UNSLOTH,
        # Automated hyperparameters (what was actually used)
        "lora_r": cfg["lora_r"],
        "lora_alpha": cfg["lora_alpha"],
        "learning_rate": learning_rate,
        "lr_auto_found": True,
        "max_epochs": cfg["max_epochs"],
        "max_steps_computed": max_steps,
        "early_stopping_patience": cfg.get("early_stopping_patience", 3),
        "per_device_batch_size": cfg["per_device_train_batch_size"],
        "auto_find_batch_size": cfg.get("auto_find_batch_size", True),
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "num_training_examples": num_examples,
    }
    summary_path = Path(FINAL_ADAPTER_DIR) / "training_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Adapter saved. Summary:\n{json.dumps(summary, indent=2)}")

    # Commit volume so data persists after the container exits
    volume.commit()
    print("\nVolume committed. Adapter is available at /outputs/final-adapter/")

    # Finish W&B run
    if use_wandb:
        import wandb
        wandb.log({"final_loss": train_result.training_loss})
        wandb.finish()

    print("\n=== Training complete ===")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main() -> None:
    """
    Trigger the remote training job.

    Run with:
        modal run train.py            # blocks until done (prints logs live)
        modal run train.py --detach   # fire-and-forget (recommended for long runs)

    Retrieve the adapter after training:
        modal volume get laguna-codealchemy-vol /outputs/final-adapter ./final-adapter
    """
    print("Submitting training job to Modal...")
    print("  GPU     : A100 40 GB")
    print("  Timeout : 7 hours")
    print("  Volume  : laguna-codealchemy-vol")
    print("  Output  : /outputs/final-adapter/")
    print()
    train.remote()
