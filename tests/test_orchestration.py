from __future__ import annotations

import json
from pathlib import Path

import pytest

from entropy_news.model.orchestration import (
    ClusterTopology,
    EnterpriseOrchestrator,
    NodeConfig,
    TrainingJob,
)
from entropy_news.utils import metrics


def test_schedule_records_plan_failure(monkeypatch) -> None:
    """Plan generation errors should increment the failure counter."""

    topology = ClusterTopology(
        nodes=[NodeConfig(name="trainer", host="127.0.0.1")]
    )
    orchestrator = EnterpriseOrchestrator(topology)
    job = TrainingJob(name="boom", entrypoint="python")

    def _boom(self, _job):  # type: ignore[override]
        raise RuntimeError("launch plan failed")

    monkeypatch.setattr(
        EnterpriseOrchestrator,
        "build_launch_plan",
        _boom,
    )

    before = metrics.REGISTRY.get_sample_value(
        "entropy_news_orchestrator_plan_failure_total",
        labels={"reason": "RuntimeError"},
    ) or 0.0

    with pytest.raises(RuntimeError):
        orchestrator.schedule(job)

    after = metrics.REGISTRY.get_sample_value(
        "entropy_news_orchestrator_plan_failure_total",
        labels={"reason": "RuntimeError"},
    ) or 0.0

    assert after >= before + 1.0
