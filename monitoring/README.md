# Monitoring Assets for Enterprise Orchestration

The files in this directory provide a baseline monitoring stack for Release 1.0
enterprise deployments. They integrate with the `EnterpriseOrchestrator`
heartbeat endpoint and the training metrics exported by the CLI workflows.

## Contents

- `prometheus.yml` – Scrape configuration for orchestrator and trainer metrics.
- `alerts.yml` – Alertmanager rules for throughput drops and gradient health
  degradation.
- `grafana_dashboard.json` – Importable Grafana dashboard showing throughput,
  gradient stability, checkpoint latency, and node health tables.

## Usage

1. Deploy Prometheus and Grafana using the Docker Compose updates or your
   preferred platform-specific tooling.
2. Mount this directory into the Prometheus container and point Grafana to the
   provided dashboard JSON via provisioning.
3. Configure exporters or sidecars on each trainer to expose metrics at port
   `8000` using the metrics emitted by `entropy-news-train`.
4. Confirm that the orchestrator's health endpoint is reachable at `/health` and
   that the custom metric `entropy_orchestrator_node_status` is being pushed via
   your telemetry agent.

These assets serve as templates; adjust scrape intervals, alert thresholds, and
panel layouts to match production requirements.
