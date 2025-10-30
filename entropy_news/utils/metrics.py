"""Metrics helpers including Prometheus instrumentation utilities."""

from __future__ import annotations

import logging
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Iterable, Optional, Tuple

try:  # pragma: no cover - exercised via integration tests
    from prometheus_client import (  # type: ignore[import-not-found]
        Counter,
        Gauge,
        Histogram,
        REGISTRY as _PROMETHEUS_REGISTRY,
        start_http_server as _start_http_server,
    )
    _PROMETHEUS_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - fallback used in minimal environments
    _PROMETHEUS_AVAILABLE = False

    class _FallbackRegistry:
        """Lightweight collector registry compatible with Prometheus scrapes."""

        def __init__(self) -> None:
            self._metrics: list["_FallbackMetric"] = []

        def register(self, metric: "_FallbackMetric") -> None:
            if metric not in self._metrics:
                self._metrics.append(metric)

        def iter_samples(self) -> Iterable[Tuple[str, Dict[str, str], float]]:
            for metric in list(self._metrics):
                yield from metric.samples()

        def get_sample_value(
            self, name: str, labels: Dict[str, str] | None = None
        ) -> float | None:
            for sample_name, sample_labels, value in self.iter_samples():
                if sample_name == name and (labels is None or labels == sample_labels):
                    return value
            return None

    class _FallbackMetric:
        """Base helper implementing label tracking for fallback metrics."""

        def __init__(
            self,
            name: str,
            documentation: str,
            labelnames: Iterable[str] = (),
            **_: object,
        ) -> None:
            self.name = name
            self.documentation = documentation
            self.labelnames = tuple(labelnames)
            self._values: Dict[Tuple[str, ...], float] = {}
            self._labels: Dict[Tuple[str, ...], Dict[str, str]] = {}
            _PROMETHEUS_REGISTRY.register(self)

        # ``labels`` accepts either positional values or keyword arguments.
        def labels(self, *label_values: str, **label_kwargs: str) -> "_MetricChild":
            if label_kwargs and label_values:
                raise ValueError("Use either positional or keyword labels, not both")
            if label_kwargs:
                values = tuple(label_kwargs[name] for name in self.labelnames)
                label_map = {name: str(label_kwargs[name]) for name in self.labelnames}
            else:
                values = tuple(label_values)
                if len(values) != len(self.labelnames):
                    raise ValueError("Incorrect number of label values supplied")
                label_map = {name: str(value) for name, value in zip(self.labelnames, values)}
            if len(values) != len(self.labelnames):
                raise ValueError("Incorrect number of label values supplied")
            key = tuple(str(v) for v in values)
            self._labels.setdefault(key, label_map)
            return self._child(key)

        def _child(self, key: Tuple[str, ...]) -> "_MetricChild":
            raise NotImplementedError

        def samples(self) -> Iterable[Tuple[str, Dict[str, str], float]]:
            raise NotImplementedError

    class _MetricChild:
        def __init__(self, metric: "_FallbackMetric", key: Tuple[str, ...]) -> None:
            self._metric = metric
            self._key = key

    class Counter(_FallbackMetric):  # type: ignore[override]
        """Minimal Counter implementation when prometheus-client is unavailable."""

        def inc(self, amount: float = 1.0) -> None:
            self._values[()] = self._values.get((), 0.0) + float(amount)

        def _child(self, key: Tuple[str, ...]) -> "_MetricChild":
            return _CounterChild(self, key)

        def _increment(self, key: Tuple[str, ...], amount: float) -> None:
            self._values[key] = self._values.get(key, 0.0) + float(amount)

        def samples(self) -> Iterable[Tuple[str, Dict[str, str], float]]:
            sample_name = f"{self.name}_total"
            for key, value in self._values.items():
                labels = self._labels.get(key, {})
                yield sample_name, labels, float(value)

    class _CounterChild(_MetricChild):
        def inc(self, amount: float = 1.0) -> None:
            self._metric._increment(self._key, amount)

    class Gauge(_FallbackMetric):  # type: ignore[override]
        """Minimal Gauge implementation when prometheus-client is unavailable."""

        def set(self, value: float) -> None:
            self._values[()] = float(value)

        def _child(self, key: Tuple[str, ...]) -> "_MetricChild":
            return _GaugeChild(self, key)

        def _set(self, key: Tuple[str, ...], value: float) -> None:
            self._values[key] = float(value)

        def samples(self) -> Iterable[Tuple[str, Dict[str, str], float]]:
            for key, value in self._values.items():
                labels = self._labels.get(key, {})
                yield self.name, labels, float(value)

    class _GaugeChild(_MetricChild):
        def set(self, value: float) -> None:
            self._metric._set(self._key, value)

    class Histogram(_FallbackMetric):  # type: ignore[override]
        """Simplified Histogram that tracks the sum of observations."""

        def observe(self, amount: float) -> None:
            self._values[()] = self._values.get((), 0.0) + float(amount)

        def _child(self, key: Tuple[str, ...]) -> "_MetricChild":
            return _HistogramChild(self, key)

        def _observe(self, key: Tuple[str, ...], amount: float) -> None:
            self._values[key] = self._values.get(key, 0.0) + float(amount)

        def samples(self) -> Iterable[Tuple[str, Dict[str, str], float]]:
            for key, value in self._values.items():
                labels = self._labels.get(key, {})
                yield self.name, labels, float(value)

    class _HistogramChild(_MetricChild):
        def observe(self, amount: float) -> None:
            self._metric._observe(self._key, amount)

    _PROMETHEUS_REGISTRY = _FallbackRegistry()

    def _start_http_server(port: int) -> HTTPServer:
        """Expose the fallback registry via a simple HTTP server."""

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # type: ignore[override]
                if self.path not in {"/", "/metrics"}:
                    self.send_response(404)
                    self.end_headers()
                    return
                lines = []
                for name, labels, value in _PROMETHEUS_REGISTRY.iter_samples():
                    label_fragment = ""
                    if labels:
                        label_fragment = "{" + ",".join(
                            f"{key}=\"{val}\"" for key, val in labels.items()
                        ) + "}"
                    lines.append(f"{name}{label_fragment} {value}")
                payload = ("\n".join(lines) + "\n").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:  # type: ignore[override]
                logger.debug("fallback_metrics: " + format, *args)

        server = HTTPServer(("", port), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

REGISTRY = _PROMETHEUS_REGISTRY
_FALLBACK_WARNING_EMITTED = False

logger = logging.getLogger(__name__)


def perplexity(entropy: float) -> float:
    """Return the perplexity associated with a cross-entropy value.

    Args:
        entropy: Cross-entropy value.

    Returns:
        float: ``math.inf`` if ``entropy`` is infinite, otherwise ``e`` raised
        to ``entropy``.
    """
    if math.isinf(entropy):
        return float("inf")
    return math.exp(entropy)


_METRICS_LOCK = threading.Lock()
_METRICS_STARTED = False
_METRICS_PORT: int | None = None


def start_metrics_server(port: int | None = None) -> int:
    """Start the Prometheus metrics HTTP exporter if it is not already running."""

    global _METRICS_STARTED, _METRICS_PORT
    with _METRICS_LOCK:
        if _METRICS_STARTED and _METRICS_PORT is not None:
            return _METRICS_PORT
        default_port = int(os.environ.get("ENTROPY_NEWS_METRICS_PORT", "8000"))
        listen_port = port or default_port
        global _FALLBACK_WARNING_EMITTED
        if not _PROMETHEUS_AVAILABLE and not _FALLBACK_WARNING_EMITTED:
            logger.warning(
                "prometheus_client not installed; using fallback metrics exporter"
            )
            _FALLBACK_WARNING_EMITTED = True
        _start_http_server(listen_port)
        _METRICS_STARTED = True
        _METRICS_PORT = listen_port
        logger.info("Prometheus metrics exporter listening on port %s", listen_port)
        return listen_port


# Training metrics -----------------------------------------------------------

_TRAINING_BATCH_SECONDS = Histogram(
    "entropy_news_training_batch_seconds",
    "Distribution of training batch durations.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)
_TRAINING_SAMPLES_TOTAL = Counter(
    "entropy_news_training_samples_total",
    "Total number of samples processed during training.",
)
_TRAINING_THROUGHPUT = Gauge(
    "entropy_news_training_throughput_samples_per_second",
    "Instantaneous throughput observed for the latest batch.",
)
_TRAINING_GRADIENT_NORM = Gauge(
    "entropy_news_training_gradient_norm",
    "L2 norm of gradients computed for the latest optimisation step.",
)
_TRAINING_EPOCH = Gauge(
    "entropy_news_training_epoch",
    "Current training epoch processed by the trainer.",
)
_TRAINING_VALIDATION_LOSS = Gauge(
    "entropy_news_training_validation_loss",
    "Validation loss observed after the latest epoch.",
)
_TRAINING_CHECKPOINT_SECONDS = Histogram(
    "entropy_news_training_checkpoint_seconds",
    "Time taken to persist model checkpoints in seconds.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)
_TRAINING_LAST_CHECKPOINT_EPOCH = Gauge(
    "entropy_news_training_last_checkpoint_epoch",
    "Epoch index associated with the most recent checkpoint event.",
)


def observe_training_batch(sample_count: int, duration_seconds: float) -> None:
    """Record throughput metrics for a processed training batch."""

    if sample_count <= 0 or duration_seconds <= 0:
        return
    _TRAINING_BATCH_SECONDS.observe(duration_seconds)
    _TRAINING_SAMPLES_TOTAL.inc(sample_count)
    throughput = sample_count / duration_seconds
    _TRAINING_THROUGHPUT.set(throughput)


def record_gradient_norm(norm: Optional[float]) -> None:
    """Update the gradient norm gauge if a valid value is provided."""

    if norm is None or math.isnan(norm) or math.isinf(norm):
        return
    _TRAINING_GRADIENT_NORM.set(norm)


def update_training_epoch(epoch: int) -> None:
    """Expose the latest training epoch as a Prometheus gauge."""

    if epoch >= 0:
        _TRAINING_EPOCH.set(epoch)


def record_validation_loss(loss: Optional[float]) -> None:
    """Record validation loss for the most recent epoch when available."""

    if loss is None or math.isnan(loss) or math.isinf(loss):
        return
    _TRAINING_VALIDATION_LOSS.set(loss)


def observe_checkpoint(duration_seconds: float, epoch: int) -> None:
    """Record checkpoint duration and the epoch it corresponds to."""

    if duration_seconds > 0:
        _TRAINING_CHECKPOINT_SECONDS.observe(duration_seconds)
    if epoch >= 0:
        _TRAINING_LAST_CHECKPOINT_EPOCH.set(epoch)


# Orchestration metrics ------------------------------------------------------

_ORCHESTRATOR_PLAN_RANKS = Gauge(
    "entropy_news_orchestrator_plan_ranks",
    "Number of ranks present in the latest generated launch plan.",
)
_ORCHESTRATOR_LAUNCH_COUNTER = Counter(
    "entropy_news_orchestrator_rank_launch_total",
    "Total distributed ranks launched by the orchestrator.",
    labelnames=("node", "role"),
)
_ORCHESTRATOR_ACTIVE_PROCESSES = Gauge(
    "entropy_news_orchestrator_active_processes",
    "Number of processes currently tracked as active by the orchestrator.",
)
_ORCHESTRATOR_HEARTBEAT_AGE = Gauge(
    "entropy_news_orchestrator_heartbeat_age_seconds",
    "Age in seconds of the latest heartbeat per node.",
    labelnames=("node",),
)
_ORCHESTRATOR_HEARTBEAT_STATUS = Gauge(
    "entropy_news_orchestrator_heartbeat_status",
    "Node health status (1 = healthy, 0 = stale) as determined by heartbeat age.",
    labelnames=("node",),
)
_ORCHESTRATOR_JOB_DURATION = Histogram(
    "entropy_news_orchestrator_job_duration_seconds",
    "Total wall-clock duration from launch until all processes exit.",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1200, 3600),
)


def record_launch_plan_size(world_size: int) -> None:
    """Expose the number of ranks in the current launch plan."""

    if world_size >= 0:
        _ORCHESTRATOR_PLAN_RANKS.set(world_size)


def record_rank_launch(node: str, role: str) -> None:
    """Increment launch counters for a specific node and role."""

    _ORCHESTRATOR_LAUNCH_COUNTER.labels(node=node, role=role).inc()


def update_active_processes(count: int) -> None:
    """Set the gauge tracking active orchestrator-managed processes."""

    if count >= 0:
        _ORCHESTRATOR_ACTIVE_PROCESSES.set(count)


def record_heartbeat_age(node: str, age_seconds: float, healthy: bool) -> None:
    """Publish the heartbeat age and health flag for ``node``."""

    _ORCHESTRATOR_HEARTBEAT_AGE.labels(node=node).set(max(age_seconds, 0.0))
    _ORCHESTRATOR_HEARTBEAT_STATUS.labels(node=node).set(1.0 if healthy else 0.0)


def observe_job_duration(duration_seconds: float) -> None:
    """Record how long it took for the orchestrated job to finish."""

    if duration_seconds > 0:
        _ORCHESTRATOR_JOB_DURATION.observe(duration_seconds)
