# laguna-codealchemy

Fine-tuning [Laguna XS 2.1](https://huggingface.co/poolside/Laguna-XS-2.1) on [CodeAlchemy](https://huggingface.co/datasets/open-alchemy/code-alchemy) (IBM, June 2026) for agentic coding.

## Why

Laguna XS 2.1 (Poolside, July 2026) is a 33B MoE model with 3B active params built for agentic coding. CodeAlchemy is IBM's newly released (June 2026) synthetic lifecycle dataset covering the full development workflow: quality rewriting, comprehension, developer tasks, multi-turn debugging, and execution traces from 1.3M runs. No model trained before June 2026 has seen this data.

## Architecture

- **Orchestrator**: Gemma 4 26B (local, Ollama)
- **Coding worker**: Laguna XS 2.1 fine-tuned (Modal endpoint)
- **Final review**: Claude Sonnet 5 / Opus 4.8 (GCP quota)

## Method

QLoRA on Modal (A100 40GB, ~$15-22, within $30 free credits)

## Datasets

- Primary: `open-alchemy/code-alchemy` (~100k subset)
- Secondary: `WaltonFuture/agentic-sft-new` (~100k subset)
- Tertiary: SWE-Fixer training data (~50k)

## Files

- `dataset_prep.py` — download and format training data
- `train.py` — Modal QLoRA training job
- `serve.py` — Modal OpenAI-compatible inference endpoint
- `configs/qlora.yaml` — training hyperparameters
