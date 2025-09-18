# Deployment Guide

This guide describes how to package and deploy the Entropy News models using the
provided Docker tooling and inference helpers.

## Build the Docker image

```bash
# From the repository root
docker build -t entropy-news:latest .
```

The image ships with the project dependencies preinstalled and uses the
`entropy_news.main_forecast` module as its default entrypoint. Override the
command to run other CLIs, for example to invoke the training workflow:

```bash
docker run --rm -v "$PWD/output":/workspace/output entropy-news:latest \
  python -m entropy_news.main --help
```

A `docker/docker-compose.yml` file is included for local iteration. It mounts the
`output/` directory so that forecasts, checkpoints, and ONNX exports persist on
the host machine.

## Quantisation and ONNX export

The `entropy_news.model.inference` module contains helpers for dynamic
quantisation and ONNX export. The snippet below demonstrates how to use them to
optimise a trained model before shipping it inside the Docker image.

```python
from pathlib import Path

import torch

from entropy_news.model import EntropyLSTM, ModelConfig
from entropy_news.model.inference import export_to_onnx, quantize_dynamic

config = ModelConfig(
    architecture="lstm",
    vocab_size=10000,
    embed_dim=100,
    hidden_dim=128,
    num_layers=2,
    num_heads=2,
    ff_dim=256,
    dropout=0.1,
)
model = EntropyLSTM(
    vocab_size=config.vocab_size,
    embed_dim=config.embed_dim,
    hidden_dim=config.hidden_dim,
)
model.load_state_dict(torch.load("output/model_final.pth", map_location="cpu"))
model.eval()

quantized = quantize_dynamic(model)
export_to_onnx(
    quantized,
    dummy_input=[0] * 32,  # Replace with a representative tokenised sequence
    path=Path("output/entropy_forecast.onnx"),
)
```

Bundle the exported assets alongside the Docker image to serve a lightweight
forecasting endpoint or batch scoring job.

## Checkpoint safety in production

When mounting checkpoints into the container, PyTorch defaults to the
``weights_only=True`` loader. This prevents executing arbitrary pickle payloads
from untrusted artifacts. If you must restore a legacy checkpoint saved via
``torch.save(model)``, pass ``--allow-unsafe-load`` to the CLI (or export
``ENTROPY_NEWS_ALLOW_UNSAFE_LOAD=1``) **after** verifying that the file comes
from a trusted source. The application logs an explicit warning whenever the
unsafe fallback is used so operators can audit deployments.

> **Note:** As the remaining legacy checkpoints are migrated to safer
> serialization formats we plan to retire the unsafe fallback entirely. Review
> custom pipelines ahead of time so the flag can be removed without
> interruptions once the default changes.

## Running inside the container

Once the assets are exported, mount them into the container and execute the
forecast CLI:

```bash
docker run --rm \
  -v "$PWD/output":/workspace/output \
  entropy-news:latest \
  --vocab-path /workspace/output/vocab.json \
  --model-path /workspace/output/model_final.pth \
  --config-path /workspace/output/model_config.json \
  --new-data /workspace/data/news_new.txt \
  --output-csv /workspace/output/forecast_results.csv
```

The quantised and ONNX-exported models can be distributed across services or
scheduled jobs to provide reproducible analytics built on the Entropy News
pipeline.
