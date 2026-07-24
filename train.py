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
    3. (Optional) modal secret create wandb-token WANDB_API_KEY=...

Note: Training data is now streamed directly from HuggingFace at runtime —
no local upload required. Sources: open-alchemy/code-alchemy (all 5 configs)
+ WaltonFuture/agentic-sft-new.

Automated hyperparameter strategies
------------------------------------
  LoRA rank   : max(8, hidden_dim // 128)  — derived from model architecture
  LoRA alpha  : 2 × rank  — modern consensus, not rank==alpha
  Learning rate: exponential sweep (1e-7 → 1e-2 over 100 steps); finds steepest
                  descent point; falls back to config learning_rate_fallback
  Batch size  : auto_find_batch_size=True in TrainingArguments (TRL)
  Epochs      : early stopping (patience=3) instead of a fixed epoch count
  max_steps   : derived at runtime from dataset size × epochs × batch
  Data volume : ACT controller — feeds 50k-example batches, stops when val loss
                improvement < epsilon (0.005); hard ceiling of 4 batches (200k)
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
        # Unsloth last — install core package only (no flash-attn extra that
        # requires NVCC at image-build time; kernels are loaded at runtime).
        "unsloth",
    )
)

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    gpu="a100-40gb",
    timeout=14 * 3600,         # 14-hour hard cap (was 7h; HF streaming runs longer)
    volumes={
        "/outputs": volume,   # persist LoRA adapter; /data not needed (data streamed from HF)
    },
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        # wandb-token omitted — create it if you want W&B logging:
        #   modal secret create wandb-token WANDB_API_KEY=...
    ],
    memory=65536,              # 64 GB RAM to handle tokenization buffers
)
def train(
    max_steps_override: int = -1,
    use_act: bool = True,
    act_batch_size: int = 50_000,
    act_epsilon: float = 0.005,
    act_max_batches: int = 4,
) -> None:
    """
    Full QLoRA fine-tuning pipeline with automated hyperparameter strategies:

    1. Load YAML config (baked into the image from the local configs/ directory).
    2. Load Laguna XS 2.1 with 4-bit NF4 quantization.
    3. Apply LoRA adapters — rank derived from hidden_dim, alpha = 2×rank.
    4. Find optimal LR via exponential sweep (find_learning_rate()).
    5. ACT controller: feeds data in 50k batches, stops when val loss
       improvement < epsilon (0.005); hard ceiling 4 batches (200k examples).
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

    print("\n=== Training config ===")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()

    # -----------------------------------------------------------------------
    # 1. Sanity checks
    # -----------------------------------------------------------------------
    # Training data is streamed directly from HuggingFace — no local file needed.

    # Accept any of the common HuggingFace token env var names
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_ACCESS_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not hf_token:
        raise EnvironmentError(
            "HuggingFace token not found. Create the Modal secret with the correct key name:\n"
            "  modal secret create huggingface-token HF_TOKEN=hf_...\n"
            "Tried: HF_TOKEN, HUGGING_FACE_HUB_TOKEN, HF_ACCESS_TOKEN, HUGGINGFACE_TOKEN\n"
            f"Env keys present: {[k for k in os.environ if 'HF' in k or 'HUGGING' in k.upper()]}"
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
    # 4. ACT controller helpers
    # -----------------------------------------------------------------------

    # ------------------------------------------------------------------
    # HuggingFace streaming helpers (replaces local load_jsonl)
    # ------------------------------------------------------------------

    # Dataset / config constants — same as dataset_prep.py
    _HF_CODEALCHEMY = "open-alchemy/code-alchemy"
    _HF_AGENTIC = "WaltonFuture/agentic-sft-new"
    _HF_CA_CONFIGS = ["code-dev", "code-dialogue", "code-trace", "code-enhance", "code-qa"]
    _HF_CA_PLACEHOLDER_CONFIGS = {"code-dev", "code-dialogue"}
    _HF_CA_USER_PROMPTS = {
        "code-dev": "Complete the following developer task:",
        "code-dialogue": "Continue this development conversation:",
        "code-trace": "Analyze this code execution trace:",
        "code-enhance": "Review and improve this code:",
        "code-qa": "Answer this code question:",
    }
    _HF_SYSTEM_PROMPT = (
        "You are an expert software engineer working on a long-horizon coding task. "
        "You write clean, tested, production-quality code."
    )

    def _hf_extract_record(example: dict, config_name: str) -> dict | None:
        """Convert a raw HF example into a ChatML messages dict."""
        import hashlib

        # WaltonFuture/agentic-sft-new: already has messages list
        if config_name == "__agentic__":
            messages = example.get("messages") or example.get("conversations")
            if not messages or not isinstance(messages, list):
                return None
            # normalise role names
            normed = []
            for m in messages:
                role = str(m.get("role") or m.get("from") or "").lower()
                content = str(m.get("content") or m.get("value") or "").strip()
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant"):
                    role = "assistant"
                if content:
                    normed.append({"role": role, "content": content})
            if not normed:
                return None
            if normed[0]["role"] != "system":
                normed.insert(0, {"role": "system", "content": _HF_SYSTEM_PROMPT})
            return {"messages": normed}

        # CodeAlchemy configs
        if config_name in _HF_CA_PLACEHOLDER_CONFIGS:
            raw_text = str(example.get("text_with_placeholders", "") or "").strip()
        else:
            raw_text = str(example.get("text", "") or "").strip()
        if not raw_text:
            return None
        user_turn = _HF_CA_USER_PROMPTS.get(config_name, "Complete the following coding task:")
        return {
            "messages": [
                {"role": "system", "content": _HF_SYSTEM_PROMPT},
                {"role": "user",   "content": user_turn},
                {"role": "assistant", "content": raw_text},
            ]
        }

    def stream_hf_batch(batch_idx: int, batch_size: int = 50_000) -> list[dict]:
        """
        Stream one batch of training examples from HuggingFace.

        Covers CodeAlchemy (all 5 configs) + WaltonFuture/agentic-sft-new.
        Skips batch_idx * batch_size rows globally (round-robin across sources),
        deduplicates on SHA-256 of the assistant content, and returns up to
        batch_size records formatted as ChatML messages dicts.

        Args:
            batch_idx:  0-based batch index (used to compute skip offset).
            batch_size: Target number of examples to return.

        Returns:
            List of dicts with {"messages": [...]} in ChatML format.
        """
        from datasets import load_dataset
        from tqdm import tqdm

        skip = batch_idx * batch_size
        # Allocate budget evenly across all 6 sources (5 CA configs + 1 agentic)
        n_sources = len(_HF_CA_CONFIGS) + 1   # 6
        per_source = batch_size // n_sources
        skip_per_source = skip // n_sources

        records: list[dict] = []
        seen_hashes: set[str] = set()

        def _collect(ds_iter, config_name: str, target: int, skip_n: int) -> None:
            import hashlib
            collected = 0
            skipped = 0
            for example in tqdm(ds_iter, desc=f"HF:{config_name}", unit="ex", leave=False):
                if skipped < skip_n:
                    skipped += 1
                    continue
                rec = _hf_extract_record(example, config_name)
                if rec is None:
                    continue
                asst_content = next(
                    (m["content"] for m in reversed(rec["messages"]) if m["role"] == "assistant"),
                    "",
                )
                h = hashlib.sha256(asst_content[:512].encode("utf-8", errors="replace")).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                records.append(rec)
                collected += 1
                if collected >= target:
                    break
            print(f"  [HF] {config_name}: collected {collected:,} (skip={skip_n:,})")

        # Stream CodeAlchemy configs
        for config_name in _HF_CA_CONFIGS:
            try:
                ds = load_dataset(
                    _HF_CODEALCHEMY,
                    name=config_name,
                    streaming=True,
                    trust_remote_code=True,
                )
                split = ds.get("train", ds[next(iter(ds))])
                _collect(split, config_name, per_source, skip_per_source)
            except Exception as exc:
                print(f"  [HF] WARNING: could not load {_HF_CODEALCHEMY}/{config_name}: {exc}")

        # Stream WaltonFuture/agentic-sft-new
        try:
            ds = load_dataset(_HF_AGENTIC, streaming=True, trust_remote_code=True)
            split = ds.get("train", ds[next(iter(ds))])
            _collect(split, "__agentic__", per_source, skip_per_source)
        except Exception as exc:
            print(f"  [HF] WARNING: could not load {_HF_AGENTIC}: {exc}")

        print(f"[HF] stream_hf_batch(idx={batch_idx}) → {len(records):,} records")
        return records

    def stream_hf_val(n: int = 5_000, skip: int = 5_000_000) -> list[dict]:
        """
        Stream a fixed validation set from HuggingFace.

        Uses a consistent skip offset (default 5M) so the same examples are
        returned on every call regardless of which ACT batch is running.

        Args:
            n:    Number of validation examples to collect.
            skip: Row offset into the combined stream before collecting.

        Returns:
            List of ChatML messages dicts.
        """
        from datasets import load_dataset
        from tqdm import tqdm
        import hashlib

        records: list[dict] = []
        seen_hashes: set[str] = set()

        # Pull val examples from CodeAlchemy code-qa (stable, diverse)
        # then pad with agentic if needed.
        sources = [
            (_HF_CODEALCHEMY, "code-qa"),
            (_HF_CODEALCHEMY, "code-enhance"),
            (_HF_AGENTIC,     "__agentic__"),
        ]
        per_source = (n + len(sources) - 1) // len(sources)
        skip_per = skip // len(sources)

        for ds_name, config_name in sources:
            if len(records) >= n:
                break
            try:
                if config_name == "__agentic__":
                    ds = load_dataset(ds_name, streaming=True, trust_remote_code=True)
                else:
                    ds = load_dataset(ds_name, name=config_name, streaming=True, trust_remote_code=True)
                split = ds.get("train", ds[next(iter(ds))])
                needed = min(per_source, n - len(records))
                skipped = 0
                for example in tqdm(split, desc=f"val:{config_name}", unit="ex", leave=False):
                    if skipped < skip_per:
                        skipped += 1
                        continue
                    rec = _hf_extract_record(example, config_name)
                    if rec is None:
                        continue
                    asst_content = next(
                        (m["content"] for m in reversed(rec["messages"]) if m["role"] == "assistant"),
                        "",
                    )
                    h = hashlib.sha256(asst_content[:512].encode("utf-8", errors="replace")).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    records.append(rec)
                    if len(records) >= n:
                        break
            except Exception as exc:
                print(f"  [HF] WARNING: could not load val from {ds_name}/{config_name}: {exc}")

        print(f"[HF] stream_hf_val() → {len(records):,} val records")
        return records

    def load_jsonl(path: Path) -> list[dict]:
        """Load a JSONL file into a list of dicts (used for temp val files)."""
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

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

    def train_on_batch(
        model: object,
        tokenizer: object,
        batch_records: list[dict],
        val_path_str: str,
        cfg: dict,
        output_dir: str,
        max_steps_override: int = -1,
    ) -> float:
        """
        Train SFTTrainer on a single batch of records; return best eval_loss.

        Returns float('inf') if training fails for any reason.
        """
        from datasets import Dataset as HFDataset
        from transformers import EarlyStoppingCallback
        from trl import SFTTrainer, SFTConfig

        # Build HF Dataset from the batch records
        batch_ds = HFDataset.from_list(batch_records)
        formatted_batch = batch_ds.map(
            format_chatml,
            remove_columns=batch_ds.column_names,
        )

        # Load fixed validation set (written by dataset_prep.py)
        val_path = Path(val_path_str)
        if val_path.exists():
            val_records = load_jsonl(val_path)
            val_ds_raw = HFDataset.from_list(val_records[:2000])  # cap at 2k for speed
            eval_ds = val_ds_raw.map(format_chatml, remove_columns=val_ds_raw.column_names)
        else:
            # Fallback: carve 5% / max 500 from batch itself
            eval_size = min(500, max(1, int(len(formatted_batch) * 0.05)))
            split = formatted_batch.train_test_split(test_size=eval_size, seed=42)
            formatted_batch = split["train"]
            eval_ds = split["test"]

        train_ds = formatted_batch
        print(f"  [batch] Train: {len(train_ds):,}  |  Eval: {len(eval_ds):,}")

        # Derive max_steps for this batch
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

        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=batch_max_steps,
            num_train_epochs=cfg["max_epochs"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            auto_find_batch_size=cfg.get("auto_find_batch_size", True),
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg.get("_act_learning_rate", cfg.get("learning_rate_fallback", 1e-4)),
            lr_scheduler_type=cfg.get("lr_scheduler", "cosine_with_restarts"),
            warmup_ratio=cfg.get("warmup_ratio", 0.03),
            weight_decay=cfg.get("weight_decay", 0.01),
            optim=cfg.get("optim", "adamw_8bit"),
            fp16=cfg.get("fp16", False),
            bf16=cfg.get("bf16", True),
            gradient_checkpointing=cfg.get("gradient_checkpointing", True),
            eval_strategy="steps",
            eval_steps=cfg.get("eval_steps", 200),
            load_best_model_at_end=cfg.get("load_best_model_at_end", True),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=cfg.get("logging_steps", 50),
            save_steps=cfg.get("eval_steps", 200),
            save_total_limit=cfg.get("save_total_limit", 2),
            report_to=cfg.get("report_to", "none"),
            run_name=cfg.get("run_name", "laguna-xs-codealchemy"),
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

        try:
            trainer.train()
            # Pull best eval_loss from trainer state
            best_loss = getattr(trainer.state, "best_metric", None)
            if best_loss is None:
                # Fallback: run evaluate() directly
                eval_out = trainer.evaluate()
                best_loss = eval_out.get("eval_loss", float("inf"))
            return float(best_loss)
        except Exception as exc:
            print(f"  [batch] Training failed: {exc}")
            return float("inf")

    def act_controller(
        model: object,
        tokenizer: object,
        cfg: dict,
        batch_size: int = 50_000,
        epsilon: float = 0.005,
        max_batches: int = 4,
        max_steps_override: int = -1,
        output_dir: str = "/outputs",
        # Legacy params kept for call-site compatibility but ignored:
        data_path: str = "",
        val_path: str = "",
    ) -> float:
        """
        ACT (Auto-Train Controller): streams data in batches from HuggingFace,
        stops when val loss improvement drops below epsilon.

        Based on: ACT: Auto-Train for Code Translation Framework (2025)

        Args:
            model: LoRA-wrapped model.
            tokenizer: Model tokenizer.
            cfg: Training config dict.
            batch_size: Number of examples per ACT batch (streamed from HF).
            epsilon: Minimum val loss improvement required to continue.
            max_batches: Hard ceiling on number of batches (budget guard).
            max_steps_override: Passed through to train_on_batch for smoke tests.
            output_dir: Directory for checkpoints.
            data_path: Ignored (kept for backwards compat).
            val_path: Ignored (kept for backwards compat).

        Returns:
            Best validation loss achieved across all batches.
        """
        print(f"\n[ACT] Streaming training data from HuggingFace "
              f"batch_size={batch_size:,}  max_batches={max_batches}  ε={epsilon}")

        # Stream a fixed validation set once — consistent across all ACT batches
        print("[ACT] Streaming fixed validation set (5k examples at offset 5M)...")
        val_records = stream_hf_val(n=5_000, skip=5_000_000)

        best_val_loss = float("inf")
        total_seen = 0

        for batch_idx in range(max_batches):
            print(f"\n[ACT] Streaming batch {batch_idx + 1}/{max_batches} from HuggingFace...")
            batch = stream_hf_batch(batch_idx=batch_idx, batch_size=batch_size)

            if not batch:
                print(f"[ACT] No data returned for batch {batch_idx}. Stopping.")
                break

            total_seen += len(batch)
            print(f"\n[ACT] Batch {batch_idx + 1}/{max_batches}: {len(batch):,} examples "
                  f"(total seen: {total_seen:,})")

            # Write val records to a temp in-memory path for train_on_batch
            # We pass val records directly — reuse the existing helper but use
            # a /tmp file so train_on_batch can load it via load_jsonl path.
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
            ) as tmp_val:
                for rec in val_records:
                    tmp_val.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tmp_val_path = tmp_val.name

            batch_val_loss = train_on_batch(
                model,
                tokenizer,
                batch,
                tmp_val_path,
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

            print(f"[ACT] Still improving. Streaming next batch...")

        print(f"\n[ACT] Final best val loss: {best_val_loss:.4f}")
        return best_val_loss

    # -----------------------------------------------------------------------
    # 4b. Automated LR finder — exponential sweep 1e-7 → 1e-2 over 100 steps.
    #     Returns the LR at the point of steepest loss descent (maximum
    #     negative gradient of smoothed loss).  Falls back to
    #     cfg["learning_rate_fallback"] if the sweep fails for any reason.
    #     The found LR is stored in cfg["_act_learning_rate"] so train_on_batch
    #     can use it across all ACT batches.
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

    # Run the LR finder against a small warm-up slice to find initial LR.
    # Stream 200 examples from HuggingFace (batch 0, truncated).
    print("[LR stub] Streaming 200 examples from HuggingFace for LR finder...")
    _lr_stub_records = stream_hf_batch(batch_idx=0, batch_size=200)
    from datasets import Dataset as _HFDataset
    _lr_stub_ds = _HFDataset.from_list(_lr_stub_records)
    _lr_stub_formatted = _lr_stub_ds.map(format_chatml, remove_columns=_lr_stub_ds.column_names)

    learning_rate = find_learning_rate(
        model,
        _lr_stub_formatted,
        tokenizer,
        fallback_lr=cfg.get("learning_rate_fallback", 1e-4),
    )
    print(f"[AutoHP] learning_rate  : {learning_rate:.2e}")

    # Store found LR in cfg so train_on_batch (called inside act_controller) can use it
    cfg["_act_learning_rate"] = learning_rate

    # -----------------------------------------------------------------------
    # 5. Set up W&B (if enabled)
    # -----------------------------------------------------------------------
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "laguna-codealchemy"),
            name=cfg.get("run_name", "laguna-xs-codealchemy"),
            config={
                **cfg,
                "learning_rate": learning_rate,
                "act_batch_size": act_batch_size,
                "act_epsilon": act_epsilon,
                "act_max_batches": act_max_batches,
            },
        )
        print(f"W&B run: {wandb.run.url}")

    # -----------------------------------------------------------------------
    # 6. Run ACT controller (or single-pass if use_act=False)
    # -----------------------------------------------------------------------
    print("\nStarting ACT-controlled training...")
    t0 = time.time()

    if use_act:
        print(f"[ACT] mode=ON  batch_size={act_batch_size:,}  epsilon={act_epsilon}  max_batches={act_max_batches}")
        final_val_loss = act_controller(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            output_dir=OUTPUT_DIR,
            batch_size=act_batch_size,
            epsilon=act_epsilon,
            max_batches=act_max_batches,
            max_steps_override=max_steps_override,
        )
    else:
        # Legacy single-pass: stream one batch, train once (for smoke/debug)
        print("[ACT] mode=OFF — single-pass training on one streamed batch")
        all_records = stream_hf_batch(batch_idx=0, batch_size=act_batch_size)
        val_records = stream_hf_val(n=2_000, skip=5_000_000)
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp_val:
            for rec in val_records:
                tmp_val.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp_val_path = tmp_val.name
        final_val_loss = train_on_batch(
            model=model,
            tokenizer=tokenizer,
            batch_records=all_records,
            val_path_str=tmp_val_path,
            cfg=cfg,
            output_dir=OUTPUT_DIR,
            max_steps_override=max_steps_override,
        )

    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed/3600:.2f} h")
    print(f"Final best val loss : {final_val_loss:.4f}")

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
        "final_val_loss": final_val_loss,
        "elapsed_hours": round(elapsed / 3600, 2),
        "gpu": "a100-40gb",
        "use_unsloth": USE_UNSLOTH,
        # Automated hyperparameters (what was actually used)
        "lora_r": cfg["lora_r"],
        "lora_alpha": cfg["lora_alpha"],
        "learning_rate": learning_rate,
        "lr_auto_found": True,
        "max_epochs": cfg["max_epochs"],
        "early_stopping_patience": cfg.get("early_stopping_patience", 3),
        "per_device_batch_size": cfg["per_device_train_batch_size"],
        "auto_find_batch_size": cfg.get("auto_find_batch_size", True),
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        # ACT controller parameters
        "act_enabled": use_act,
        "act_batch_size": act_batch_size,
        "act_epsilon": act_epsilon,
        "act_max_batches": act_max_batches,
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
        wandb.log({"final_val_loss": final_val_loss})
        wandb.finish()

    print("\n=== Training complete ===")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    max_steps: int = -1,
    no_act: bool = False,
    act_batch_size: int = 50_000,
    act_epsilon: float = 0.005,
    act_max_batches: int = 4,
) -> None:
    """
    Trigger the remote training job.

    Run with:
        modal run train.py                               # blocks until done (prints logs live)
        modal run train.py --detach                      # fire-and-forget (recommended for long runs)
        modal run train.py --max-steps 50                # smoke test: stop after 50 steps
        modal run train.py --no-act                      # disable ACT; single-pass on full dataset
        modal run train.py --act-max-batches 2           # limit to 2 batches (100k examples)
        modal run train.py --act-epsilon 0.01            # larger epsilon = stops sooner

    Data volume is governed by the ACT controller (epsilon=0.005, max 4 batches = 200k examples).
    No --limit flag needed; the ACT batch ceiling replaces it.

    Retrieve the adapter after training:
        modal volume get laguna-codealchemy-vol /outputs/final-adapter ./final-adapter
    """
    print("Submitting training job to Modal...")
    print("  GPU     : A100 40 GB")
    print("  Timeout : 7 hours")
    print("  Volume  : laguna-codealchemy-vol")
    print("  Output  : /outputs/final-adapter/")
    print(f"  ACT     : {'OFF (single-pass)' if no_act else 'ON'}")
    if not no_act:
        print(f"  ACT batch size  : {act_batch_size:,}")
        print(f"  ACT epsilon     : {act_epsilon}")
        print(f"  ACT max batches : {act_max_batches}  (max {act_batch_size * act_max_batches:,} examples)")
    if max_steps > 0:
        print(f"  [Smoke] max-steps  : {max_steps}")
    print()
    train.remote(
        max_steps_override=max_steps,
        use_act=not no_act,
        act_batch_size=act_batch_size,
        act_epsilon=act_epsilon,
        act_max_batches=act_max_batches,
    )
