# ft_slm — Sequential Fine-Tuning of Small Language Models

A research project exploring **continual learning** strategies for small LLMs, comparing three approaches to mitigate catastrophic forgetting during sequential fine-tuning on ArXiv papers.

---

## Overview

A small LLM (Qwen2-0.5B) is fine-tuned sequentially on 4 quarterly batches of ArXiv papers (cs.LG + cs.CL, 2024). After each batch, perplexity is evaluated on all previously seen batches to measure forgetting.

Three strategies are compared:

| Strategy          | Description                                          |
| ----------------- | ---------------------------------------------------- |
| **Naive**         | Standard fine-tuning, no forgetting protection       |
| **LoRA + Replay** | LoRA adapters + 20% replay of previous batches       |
| **EWC**           | Elastic Weight Consolidation (Fisher regularization) |

### Key Results

Perplexity on batch_01 after training on all 4 batches:

| Strategy      | Perplexity (↓ better) | Forgetting |
| ------------- | --------------------- | ---------- |
| Naive         | 5.14                  | +102%      |
| EWC           | 5.02                  | +93%       |
| LoRA + Replay | **3.66**              | **-4%**    |

LoRA + Replay is the clear winner. Perplexity stays stable across all batches while Naive and EWC degrade significantly.

---

## Project Structure

```
ft_slm/
├── data/
│   └── raw/          # ArXiv papers by batch (JSON)
├── models/           # Model checkpoints
├── notebooks/
│   └── explore_data.ipynb   # Data exploration + results visualization
├── results/          # Perplexity results (JSON)
├── scripts/
│   ├── fetch_data.py        # ArXiv data collection
│   ├── train_naive.py       # Naive sequential fine-tuning
│   ├── train_lora_replay.py # LoRA + Replay strategy
│   └── train_ewc.py         # EWC strategy
├── config.yaml       # Centralized hyperparameters
├── Dockerfile        # Reproducible environment (nvidia/cuda:12.8.1)
└── pyproject.toml    # Python dependencies (uv)
```

---

## Setup

### Requirements

- Python 3.12+
- CUDA 12.8+ (tested on RTX 5080 with CUDA 13.1)
- [uv](https://github.com/astral-sh/uv)

### Install

```bash
uv sync
```

### Fetch Data

Downloads 500 ArXiv papers per quarterly batch (cs.LG + cs.CL, 2024):

```bash
uv run python scripts/fetch_data.py
```

---

## Usage

Run each strategy sequentially:

```bash
# Baseline
uv run python scripts/train_naive.py

# LoRA + Replay
uv run python scripts/train_lora_replay.py

# EWC
uv run python scripts/train_ewc.py
```

Results are saved to `results/` as JSON files. Open `notebooks/explore_data.ipynb` to visualize comparative results.

---

## Configuration

All hyperparameters are centralized in `config.yaml`:

```yaml
model:
  name: "Qwen/Qwen2-0.5B"
  max_length: 512

training:
  epochs_per_batch: 1
  batch_size: 8
  learning_rate: 0.0002

lora:
  r: 8
  alpha: 32
  dropout: 0.05

ewc:
  lambda: 0.4

data:
  replay_ratio: 0.2
  papers_per_batch: 500
```

---

## Docker

For reproducibility on Linux servers:

```bash
docker build -t ft-slm .
docker run --gpus all \
  -v ./data:/app/data \
  -v ./models:/app/models \
  -v ./results:/app/results \
  ft-slm
```

---

## Discussion

**Why does LoRA + Replay outperform EWC?**

Our EWC implementation uses only the previous batch's Fisher matrix (not accumulated), which limits its protection to one step back. A full EWC implementation would accumulate Fisher across all batches. Additionally, λ=0.4 may be too weak a regularization signal.

LoRA + Replay benefits from two complementary mechanisms: LoRA restricts weight updates to low-rank subspaces (less destructive), while replay directly re-exposes the model to old data. This redundancy makes it robustly resistant to forgetting.
