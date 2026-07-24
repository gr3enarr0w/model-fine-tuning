# model-fine-tuning

Fine-tuning experiments for local-first AI coding infrastructure.

## Models

### Laguna XS 2.1 — Full-stack agentic coder (Modal)
- Base: poolside/Laguna-XS-2.1 (33B MoE, 3B active params)
- Fine-tuned on: CodeAlchemy (IBM) + WaltonFuture/agentic-sft-new
- Runs on: Modal A100 40GB (~$15-22/run, within $30 free credits)
- Role: complex/multi-language/long-horizon coding tasks

### Gemma 4 E4B — Language-specialized coders (local, M2 Pro)
- Base: google/gemma-4-e4b-it (~4B effective params via MoE)
- **Two strategies benchmarked head-to-head:**
  - **Strategy A — Per-language**: one model per language, trained on language-filtered CodeAlchemy subset
  - **Strategy B — Generalist**: single model trained across all languages
- Runs on: Ollama locally (M2 Pro 32GB, fast inference, runs multiple simultaneously)
- Role: fast cheap language-specific coding routed by orchestrator

## Architecture

```
Gemma 4 26B (orchestrator, local Ollama)
    ↓ detects language → routes
Gemma 4 E4B Python-ft  ← M2 Pro local
Gemma 4 E4B Go-ft      ← M2 Pro local
Gemma 4 E4B TS-ft      ← M2 Pro local
Gemma 4 E4B Java-ft    ← M2 Pro local
    ↓ complex/multi-language
Laguna XS 2.1          ← Modal endpoint
    ↓ final review
Claude Sonnet 5 / Opus 4.8  ← GCP $500 quota
```

## Benchmark Design (Strategy A vs B)

Evaluate per language: HumanEval, multi-turn debugging, code review quality, inference latency on M2 Pro.

## Datasets

- `open-alchemy/code-alchemy` — IBM CodeAlchemy (June 2026, ~976B tokens)
- `WaltonFuture/agentic-sft-new` — agentic tool use (711k examples)

## Files

| File | Purpose |
|---|---|
| `dataset_prep.py` | Download CodeAlchemy + WaltonFuture subsets |
| `train.py` | Laguna XS 2.1 QLoRA on Modal |
| `train_gemma.py` | Gemma E4B LoRA — both strategies |
| `serve.py` | Laguna Modal OpenAI-compatible endpoint |
| `benchmark.py` | Strategy A vs B evaluation per language |
| `upload_to_modal.py` | Upload local datasets/artifacts to Modal volume |
| `download_from_modal.py` | Download trained adapters from Modal volume |
| `merge_adapters.py` | Merge LoRA adapter weights into base model for deployment |
| `push_to_ollama.py` | Convert merged model to GGUF and register with Ollama |
| `run_all_parallel.py` | Launch all per-language Gemma fine-tunes in parallel |
| `configs/qlora.yaml` | Laguna QLoRA hyperparameters |
| `configs/lora_gemma.yaml` | Gemma hyperparameters |

## Getting Started

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure Modal (for cloud training)
modal setup

# 4. (Optional) Log in to Hugging Face for gated models
huggingface-cli login

# 5. Prepare datasets
python dataset_prep.py

# 6. Train (choose one)
python train.py                          # Laguna on Modal
python train_gemma.py --strategy per-lang  # Gemma per-language on Modal
python train_gemma.py --strategy generalist

# 7. Download trained adapters from Modal
python download_from_modal.py --model gemma-python

# 8. Merge adapter into base model
python merge_adapters.py --model gemma-python --output merged/gemma-python

# 9. Convert to GGUF and push to Ollama
python push_to_ollama.py --model gemma-python --quantize q5_k_m

# 10. Benchmark strategies
python benchmark.py
```
