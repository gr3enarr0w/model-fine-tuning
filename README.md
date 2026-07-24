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
| `configs/qlora_laguna.yaml` | Laguna hyperparameters |
| `configs/lora_gemma.yaml` | Gemma hyperparameters |
