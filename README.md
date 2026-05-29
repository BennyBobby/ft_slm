# ft_slm : Sequential Fine-Tuning of Small Language Models

A research project exploring **continual learning** strategies for small LLMs, comparing five approaches to mitigate catastrophic forgetting during sequential fine-tuning on ArXiv papers.

---

## Overview

A small LLM (Qwen2-0.5B) is fine-tuned sequentially on 4 quarterly batches of ArXiv papers (cs.LG + cs.CL, 2024). After each batch, perplexity is evaluated on all previously seen batches to measure forgetting.

Five strategies are compared:

| Strategy                  | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| **Naive**                 | Standard fine-tuning, no forgetting protection               |
| **EWC (λ=0.4)**           | Elastic Weight Consolidation, single-step Fisher             |
| **EWC accumulated (λ=5)** | EWC with accumulated Fisher across all batches, stronger λ   |
| **LoRA + Replay**         | LoRA adapters + 20% replay of previous batches               |
| **O-LoRA**                | Orthogonal LoRA, penalizes overlap between adapter subspaces |

### Key Results

Perplexity on batch_01 after training on all 4 batches (lower = less forgetting):

| Strategy                | Perplexity (↓ better) | vs. initial |
| ----------------------- | --------------------- | ----------- |
| Naive                   | 5.14                  | +102%       |
| EWC (λ=0.4)             | ~5.02                 | +97%        |
| EWC accumulated (λ=5.0) | 5.05                  | +98%        |
| **O-LoRA**              | **3.70**              | **+45%**    |
| **LoRA + Replay**       | **3.66**              | **+43%**    |

**O-LoRA matches LoRA + Replay** without storing any old data. There a significant result, as it shows orthogonal subspace adaptation alone is sufficient to prevent most forgetting in this domain-specific setting.

EWC consistently underperforms, even with Fisher accumulation and stronger regularization, likely because the intra-domain similarity between batches makes gradient² importance estimates unreliable.

### Benchmark Results (LoRA + Replay checkpoints)

Evaluated on ARC-Easy, HellaSwag, and PIQA (zero-shot) using [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness):

| Task      | Base model | After batch_04 | Delta |
| --------- | ---------- | -------------- | ----- |
| ARC-Easy  | ~48%       | ~50%           | +2.2% |
| HellaSwag | ~59%       | ~59%           | ≈ 0%  |
| PIQA      | ~72%       | ~71%           | −1.2% |

Fine-tuning on domain-specific text slightly improves factual reasoning (ARC-Easy), leaves commonsense inference unchanged (HellaSwag), and has negligible impact on physical intuition (PIQA).

---

## Project Structure

```
ft_slm/
├── data/
│   └── raw/                    # ArXiv papers by batch (JSON)
├── models/
│   └── lora_replay/            # Merged checkpoints per batch
├── notebooks/
│   ├── explore_data.ipynb      # Data exploration (distributions, samples)
│   └── results.ipynb           # Result visualizations (all 5 strategies)
├── results/                    # Perplexity + benchmark results (JSON)
├── scripts/
│   ├── fetch_data.py           # ArXiv data collection
│   ├── utils.py                # Shared utilities (model loading, dataset, perplexity)
│   ├── train_naive.py          # Naive sequential fine-tuning
│   ├── train_lora_replay.py    # LoRA + Replay strategy
│   ├── train_ewc.py            # EWC with accumulated Fisher (λ=5.0)
│   ├── train_olora.py          # O-LoRA: orthogonal subspace adaptation
│   └── eval_benchmarks.py      # Benchmark evaluation (arc_easy, hellaswag, piqa)
├── config.yaml                 # Centralized hyperparameters
├── Dockerfile                  # Reproducible environment (nvidia/cuda:12.8.1)
├── pyproject.toml              # Python dependencies (uv)
├── report.tex                  # LaTeX source of the project report
└── report.pdf                  # Compiled report (PDF)
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

Run each strategy independently:

```bash
# Baseline
uv run python scripts/train_naive.py

# LoRA + Replay (saves checkpoints to models/lora_replay/)
uv run python scripts/train_lora_replay.py

# EWC with accumulated Fisher
uv run python scripts/train_ewc.py

# O-LoRA
uv run python scripts/train_olora.py
```

Evaluate LoRA+Replay checkpoints on standard benchmarks:

```bash
uv run python scripts/eval_benchmarks.py
```

Results are saved to `results/` as JSON files. Open `notebooks/results.ipynb` to visualize all strategies.

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
  lambda: 5.0

olora:
  lambda: 1.0

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

### Why does EWC underperform?

EWC uses the Fisher information matrix to identify important parameters and penalizes deviation from them. In practice, when batches come from the same domain (ML papers from the same year), gradient magnitudes are similar across tasks. Fisher estimates don't cleanly separate "important for batch_01" from "important for batch_02". The penalty ends up being too diffuse to prevent forgetting. Even with Fisher accumulation (λ=5.0), performance matches the Naive baseline.

### Why does O-LoRA match LoRA + Replay?

O-LoRA forces each new batch's LoRA A matrices to be orthogonal to those of previous batches, ensuring updates live in different subspaces of the weight space. This prevents new learning from overwriting old representations without storing any data.

The result (3.70 vs 3.66 perplexity) suggests that for domain-specific continual fine-tuning, structural orthogonality is as effective as data replay and more memory-efficient.

### Why does LoRA help in both cases?

LoRA restricts updates to a low-rank subspace (r=8 out of 512 hidden dims). This inherently limits the destructive capacity of each update, making forgetting less severe even before adding orthogonality or replay constraints.
