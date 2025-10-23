# Inference Delivery Tutorial

Deploy trained models to production endpoints with reproducible packaging and
observability hooks.

## Objectives

1. Package models for multiple runtimes.
2. Integrate monitoring and rollback strategies.
3. Validate performance prior to promotion.

## 1. Build the Inference Image

Use the Dockerfile and Helm overrides published in :doc:`../runbooks/enterprise_training`
for a reference container.

```bash
docker build -t entropy-news-inference -f Dockerfile .
```

For Kubernetes clusters, apply the Helm overrides in
`docs/runbooks/helm-values.yaml` to align resource limits with production
expectations.

## 2. Export Model Artifacts

```bash
python - <<'PY'
import torch
from entropy_news.model import ModelFactory, ModelConfig
from entropy_news.model.inference import export_to_onnx

config = ModelConfig.load("configs/resolved.json")
model = ModelFactory.create(config, embedding_matrix=None)
model.load_state_dict(torch.load("output/checkpoints/latest.pt"))
model.eval()

export_to_onnx(model, dummy_input=[0] * 100, path="dist/model.onnx")
torch.jit.script(model).save("dist/model.ts")
print("Artifacts written to dist/model.onnx and dist/model.ts")
PY
```

Store the exported artefacts in object storage and register the bundle in the
model registry defined in :doc:`../playbooks/research_registry`.

## 3. Wire Monitoring and Alerts

- Enable the Prometheus scrape configuration found in `monitoring/prometheus.yml`.
- Import the Grafana dashboard from `monitoring/grafana_dashboard.json` and map
  the inference latency panels to your environment.
- Configure alert thresholds as outlined in `monitoring/alerts.yml` and test
  them using the chaos scripts in :doc:`../playbooks/incident_response`.

## 4. Run Pre-production Checks

Execute the smoke tests and replay harness:

```bash
pytest tests/test_integration_workflows.py -k "forecast"
entropy-news-forecast --model-path output/checkpoints/latest.pt \
  --vocab-path output/vocab.json \
  --config-path configs/resolved.json \
  --new-data data/canary.csv \
  --output-csv output/canary_metrics.csv
```

Promote the release once the tests pass and monitoring baselines stabilise for a
24-hour window. Capture the rollout in the :doc:`../playbooks/enterprise_rollout`
logbook for auditability.
