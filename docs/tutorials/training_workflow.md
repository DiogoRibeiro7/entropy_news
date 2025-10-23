# Training Workflow Tutorial

This tutorial covers how to train an entropy forecasting model from scratch
using the packaged configuration utilities.

## Prerequisites

- Install the project with the `dev` extras: `pip install -e .[dev]`.
- Download the sample dataset from `data/sample_headlines.csv` or supply your
  own CSV with `headline`, `timestamp`, and `label` columns.
- Review the configuration dataclass in
  {mod}`entropy_news.model.config.ModelConfig` to understand available
  parameters.

## 1. Create a Training Configuration

Start by generating a baseline configuration file with the
{class}`entropy_news.model.config.ModelConfig` dataclass. The snippet below writes
a JSON payload that mirrors the repository defaults.

```bash
python - <<'PY'
from entropy_news.model.config import ModelConfig

config = ModelConfig(
    architecture="lstm",
    vocab_size=10000,
    embed_dim=100,
    hidden_dim=128,
    num_heads=2,
    ff_dim=256,
    num_layers=2,
    dropout=0.1,
)
config.validate()
config.save("configs/baseline.json")
print("Saved configs/baseline.json")
PY
```

Edit the generated file to match your dataset paths and hyperparameter goals.
Use inline comments in version control or accompanying notes to capture the
rationale behind each change so future audits are straightforward.

## 2. Launch the Trainer

Kick off training with the CLI wrapper. This command streams logs to the
console, persists checkpoints when requested, and writes the resolved
configuration for later reuse.

```bash
entropy-news-train --model-config configs/baseline.json --epochs 10 \
  --train-data data/sample_headlines.csv \
  --checkpoint output/checkpoints/latest.pt \
  --config-out configs/resolved.json
```

Behind the scenes the CLI delegates to
{mod}`entropy_news.model.factory.ModelFactory` and the
{class}`entropy_news.model.trainer.Trainer` to assemble datasets, models, and
optimisers. Review the API reference for deeper insights into each component.

## 3. Monitor Progress

- Tail the streaming logs by redirecting output to a file or piping through
  tools like `tee` for collaborative reviews.
- Visualise gradients and learning rates by running the dashboard in monitoring
  mode: `streamlit run entropy_news/dashboard/app.py -- --metrics output/training/metrics.json`.
- Use the orchestration health checks documented in
  {mod}`entropy_news.model.orchestration` when training across multiple nodes.

## 4. Validate and Export

After training converges, generate evaluation artefacts with the evaluation CLI
and export a portable model bundle.

```bash
entropy-news-eval --model-path output/checkpoints/latest.pt \
  --vocab-path output/vocab.json \
  --config-path configs/baseline.json \
  --data data/validation.csv \
  --output-csv output/evaluation/report.csv
```

Export an ONNX artefact using a short Python helper so you can control the dummy
input shape explicitly.

```bash
python - <<'PY'
from entropy_news.model import ModelFactory, ModelConfig
from entropy_news.model.inference import export_to_onnx
import torch

config = ModelConfig.load("configs/baseline.json")
model = ModelFactory.create(config, embedding_matrix=None)
model.load_state_dict(torch.load("output/checkpoints/latest.pt"))
export_to_onnx(model, dummy_input=[0] * 100, path="output/model.onnx")
print("Exported output/model.onnx")
PY
```

The evaluation report summarises accuracy, calibration, and entropy drift. The
ONNX export enables deployment to low-latency inference environments.

## 5. Next Steps

- Schedule recurring trainings with the enterprise orchestrator and consult the
  :doc:`../playbooks/enterprise_rollout` playbook for rollout guidance.
- Extend the tutorial by layering the causal toolkit described in
  :doc:`../tutorials/forecasting_insights` to analyse counterfactual outcomes.
