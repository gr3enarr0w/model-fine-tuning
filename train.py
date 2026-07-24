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
    Full QLoRA fine-tuning pipeline:

    1. Load YAML config from /configs/qlora.yaml (baked into the image from
       the local configs/ directory at build time — see note below).
    2. Load Laguna XS 2.1 with 4-bit NF4 quantization.
    3. Apply LoRA adapters via Unsloth (2× faster on A100 vs vanilla PEFT).
    4. Load training data from /data/train.jsonl.
    5. Run SFTTrainer from TRL.
    6. Save the LoRA adapter to /outputs/final-adapter/.
    7. Commit the volume so the adapter persists after the container exits.
    """

    # -----------------------------------------------------------------------
    # Lazy imports (all inside the Modal function so they resolve in the image)
    # -----------------------------------------------------------------------
    import json
    import os
    import sys
    import time
    from pathlib import Path

    import torch
    import yaml

    # -----------------------------------------------------------------------
    # 0. Read config
    # -----------------------------------------------------------------------
    CONFIG_PATH = Path("/configs/qlora.yaml")

    # Fallback inline defaults if the config file was not uploaded.
    DEFAULTS: dict = {
        "model_name": "poolside/Laguna-XS-2.1",
        "max_seq_length": 4096,
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,   # effective batch = 16
        "num_train_epochs": 1,
        "max_steps": -1,                     # -1 = full epoch
        "warmup_ratio": 0.03,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "logging_steps": 50,
        "save_steps": 500,
        "save_total_limit": 2,
        "fp16": False,
        "bf16": True,                        # A100 supports BF16 natively
        "optim": "adamw_8bit",
        "gradient_checkpointing": True,
        "packing": True,                     # sequence packing for efficiency
        "dataset_text_field": None,          # we use a formatting_func instead
        "output_dir": "/outputs/checkpoints",
        "run_name": "laguna-xs-codealchemy",
        "report_to": "wandb",                # set to "none" if no WANDB key
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

    print("\n=== Training config ===")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()

    MODEL_NAME: str = cfg["model_name"]
    OUTPUT_DIR: str = cfg["output_dir"]
    FINAL_ADAPTER_DIR = "/outputs/final-adapter"
    DATA_PATH = Path("/data/train.jsonl")

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
    print("\nApplying LoRA adapters...")

    if USE_UNSLOTH:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"],
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
            target_modules=cfg["target_modules"],
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
    print(f"Loaded {len(raw_ds):,} training examples.")

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

    # -----------------------------------------------------------------------
    # 5. Set up W&B (if enabled)
    # -----------------------------------------------------------------------
    if use_wandb:
        import wandb
        wandb.init(
            project="laguna-codealchemy",
            name=cfg["run_name"],
            config=cfg,
        )
        print(f"W&B run: {wandb.run.url}")

    # -----------------------------------------------------------------------
    # 6. Configure and run SFTTrainer
    # -----------------------------------------------------------------------
    from trl import SFTTrainer, SFTConfig

    print("\nStarting SFT training...")
    t0 = time.time()

    # Build TrainingArguments-equivalent via SFTConfig
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        optim=cfg["optim"],
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        report_to=cfg["report_to"],
        run_name=cfg["run_name"],
        # SFT-specific
        max_seq_length=cfg["max_seq_length"],
        packing=cfg["packing"],
        dataset_text_field="text",
        # Disable dataset splitting (we manage our own eval set)
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_ds,
        args=sft_config,
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

    if USE_UNSLOTH:
        # Unsloth provides a convenience saver
        model.save_pretrained(FINAL_ADAPTER_DIR)
        tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
    else:
        model.save_pretrained(FINAL_ADAPTER_DIR)
        tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

    # Write a small training summary alongside the adapter
    summary = {
        "model_name": MODEL_NAME,
        "run_name": cfg["run_name"],
        "final_loss": train_result.training_loss,
        "global_step": train_result.global_step,
        "elapsed_hours": round(elapsed / 3600, 2),
        "gpu": "a100-40gb",
        "use_unsloth": USE_UNSLOTH,
        "lora_r": cfg["lora_r"],
        "lora_alpha": cfg["lora_alpha"],
        "learning_rate": cfg["learning_rate"],
        "num_train_epochs": cfg["num_train_epochs"],
        "per_device_batch_size": cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
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
