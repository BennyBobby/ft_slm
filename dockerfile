FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .python-version ./

RUN uv sync --frozen

COPY . .

CMD ["uv", "run", "python", "scripts/train.py"]
