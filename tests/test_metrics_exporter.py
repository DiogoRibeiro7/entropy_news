"""Tests for Prometheus metrics helpers."""

from __future__ import annotations

from entropy_news.utils import metrics

REGISTRY = metrics.REGISTRY


def test_start_metrics_server_runs_once(monkeypatch) -> None:
    """Starting the exporter twice should only invoke the HTTP server once."""

    calls: list[int] = []

    class _FakeServer:
        def __init__(self, port: int) -> None:
            self.port = port
            self.shutdown_called = False

        def shutdown(self) -> None:
            self.shutdown_called = True

    server = _FakeServer(0)

    def _fake_start(port: int) -> _FakeServer:
        calls.append(port)
        server.port = port
        return server

    monkeypatch.setattr(metrics, "_start_http_server", _fake_start)

    first = metrics.start_metrics_server(9200)
    second = metrics.start_metrics_server(9300)

    assert calls == [9200]
    assert first is second
    assert first.port == 9200
    assert callable(first.shutdown)

    metrics.stop_metrics_server()
    assert server.shutdown_called is True


def test_training_metrics_update_throughput() -> None:
    """Training helpers should set throughput, samples, and gradient metrics."""

    metrics.observe_training_batch(256, 2.0)
    metrics.record_gradient_norm(12.5)
    metrics.observe_checkpoint(0.25, 3)

    throughput = REGISTRY.get_sample_value(
        "entropy_news_training_throughput_samples_per_second"
    )
    samples = REGISTRY.get_sample_value("entropy_news_training_samples_total")
    gradient_norm = REGISTRY.get_sample_value("entropy_news_training_gradient_norm")
    checkpoint_epoch = REGISTRY.get_sample_value(
        "entropy_news_training_last_checkpoint_epoch"
    )

    sum_before = REGISTRY.get_sample_value("entropy_news_training_batch_seconds_sum") or 0.0
    count_before = (
        REGISTRY.get_sample_value("entropy_news_training_batch_seconds_count") or 0.0
    )
    bucket_before = REGISTRY.get_sample_value(
        "entropy_news_training_batch_seconds_bucket",
        labels={"le": "1.0"},
    ) or 0.0

    metrics.observe_training_batch(128, 1.0)

    sum_after = REGISTRY.get_sample_value("entropy_news_training_batch_seconds_sum") or 0.0
    count_after = (
        REGISTRY.get_sample_value("entropy_news_training_batch_seconds_count") or 0.0
    )
    bucket_after = REGISTRY.get_sample_value(
        "entropy_news_training_batch_seconds_bucket",
        labels={"le": "1.0"},
    ) or 0.0

    assert throughput == 128.0
    assert samples and samples >= 256.0
    assert gradient_norm == 12.5
    assert checkpoint_epoch == 3.0
    assert sum_after >= sum_before + 1.0
    assert count_after >= count_before + 1.0
    assert bucket_after >= bucket_before + 1.0


def test_orchestrator_metrics_capture_labels() -> None:
    """Orchestrator metrics should register launch counts and heartbeat age."""

    metrics.record_launch_plan_size(4)
    metrics.record_rank_launch("trainer-a", "trainer")
    metrics.record_rank_launch("trainer-a", "trainer")
    metrics.update_active_processes(2)
    metrics.record_heartbeat_age("trainer-a", 1.5, True)

    launches = REGISTRY.get_sample_value(
        "entropy_news_orchestrator_rank_launch_total",
        labels={"node": "trainer-a", "role": "trainer"},
    )
    plan = REGISTRY.get_sample_value("entropy_news_orchestrator_plan_ranks")
    active = REGISTRY.get_sample_value("entropy_news_orchestrator_active_processes")
    heartbeat = REGISTRY.get_sample_value(
        "entropy_news_orchestrator_heartbeat_age_seconds",
        labels={"node": "trainer-a"},
    )

    assert launches == 2.0
    assert plan == 4.0
    assert active == 2.0
    assert heartbeat == 1.5
