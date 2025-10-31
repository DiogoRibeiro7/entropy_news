# Prometheus Metrics Exporter Review

## Executive Summary
- Training and orchestration workflows now expose Prometheus counters, gauges, and histograms, satisfying the Release 1.0 monitoring requirement for throughput, gradient health, and checkpoint timings.
- CLI flags (`--enable-metrics`, `--metrics-port`) exist for both training and orchestration entrypoints so operators can opt in to exporters without code changes.
- Documentation and monitoring assets describe how to scrape the new endpoints, and regression tests exercise the instrumentation helpers.

## Roadmap Compliance
- **Implementation plan alignment:** Release 1.0 Phase 3 calls for Prometheus collectors targeting throughput, gradient health, and checkpoint timings; the new helpers in `entropy_news/utils/metrics.py` emit exactly those signals for trainers and orchestrator jobs.
- **CLI enablement:** `entropy_news/main.py` and `entropy_news/model/orchestration.py` add `--enable-metrics` flags to start exporters, meeting the requirement that operators can toggle metrics from the toolchain.
- **Documentation:** The README and `monitoring/README.md` describe exporter usage, ensuring the monitoring stack in `docker-compose` can scrape the metrics endpoints.
- **Testing:** `tests/test_metrics_exporter.py` covers server startup idempotency, training throughput counters, and orchestration label handling.

## Findings
1. **Fallback histogram semantics differ from Prometheus expectations**  
   - The fallback `Histogram` implementation only accumulates sums and does not produce `_bucket`, `_sum`, and `_count` samples. Consumers relying on histogram quantiles in environments without `prometheus_client` will see incomplete data.  
   - **Recommendation:** Expand the fallback to expose Prometheus-compatible buckets and counters (or require `prometheus_client`).

2. **Exporter lifecycle lacks shutdown hooks**  
   - `start_metrics_server` launches a daemon HTTP server but never returns a handle to stop it. For long-lived tests or embedded integrations this can leave stray threads.  
   - **Recommendation:** Provide an optional shutdown function for the fallback server and document cleanup expectations for the official client.

3. **Orchestrator metrics assume plan builds succeed**  
   - When `build_launch_plan` raises, no telemetry is recorded. Consider emitting failure counters to aid alerting on orchestration errors.  
   - **Recommendation:** Add exception handling around schedule/launch operations that records failure metrics before surfacing the error.

