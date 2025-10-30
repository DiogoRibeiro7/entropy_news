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
3. Launch training jobs with ``entropy-news-train --enable-metrics`` (optionally
   ``--metrics-port``) so Prometheus can scrape throughput, gradient norms, and
   checkpoint latency at the configured port (`8000` by default).
4. Run the orchestrator with ``entropy-news-orchestrate --enable-metrics`` and a
   ``--metrics-port`` that matches the scrape target (``9100`` in the default
   configuration). Prometheus collects launch counters, heartbeat ages, and
   active process gauges directly from this endpoint.

These assets serve as templates; adjust scrape intervals, alert thresholds, and
panel layouts to match production requirements.
