"""Tests for the enterprise orchestration layer."""

from __future__ import annotations

import json
import sys
import time
from urllib import request

import pytest

from entropy_news.model.orchestration import (
    ClusterTopology,
    EnterpriseOrchestrator,
    LaunchSpec,
    NodeConfig,
    TrainingJob,
    main,
)
from entropy_news.utils import metrics


def test_cluster_validation_requires_trainer() -> None:
    topology = ClusterTopology(
        nodes=[NodeConfig(name="monitor", host="10.0.0.1", role="monitor")],
    )
    with pytest.raises(ValueError):
        topology.validate()


def test_build_launch_plan_assigns_ranks_and_env(tmp_path) -> None:
    topology = ClusterTopology(
        nodes=[
            NodeConfig(name="trainer-a", host="10.0.0.1", processes=2),
            NodeConfig(name="trainer-b", host="10.0.0.2", processes=1),
            NodeConfig(name="eval", host="10.0.0.3", role="evaluator"),
        ],
        shared_storage=tmp_path,
        environment={"GLOBAL_FLAG": "1"},
    )
    orchestrator = EnterpriseOrchestrator(topology)
    job = TrainingJob(name="baseline", entrypoint="python", args=("-m", "train"))
    plan = orchestrator.build_launch_plan(job)
    assert len(plan) == 3
    assert {spec.rank for spec in plan} == {0, 1, 2}
    assert all(spec.env["WORLD_SIZE"] == "3" for spec in plan)
    first = plan[0]
    assert first.env["MASTER_ADDR"] == "10.0.0.1"
    assert first.env["GLOBAL_FLAG"] == "1"
    assert "ENTROPY_NEWS_CHECKPOINT_DIR" in first.env


def test_health_server_exposes_report(tmp_path) -> None:
    topology = ClusterTopology(
        nodes=[NodeConfig(name="trainer", host="127.0.0.1")],
        shared_storage=tmp_path,
    )
    orchestrator = EnterpriseOrchestrator(topology, health_timeout=10.0)
    orchestrator.register_heartbeat("trainer")
    port = orchestrator.start_health_server(port=0)
    try:
        time.sleep(0.1)
        with request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            payload = response.read().decode("utf-8")
        report = json.loads(payload)
        assert report["trainer"]["status"] == "healthy"
        assert report["trainer"]["role"] == "trainer"
    finally:
        orchestrator.stop_health_server()


def test_schedule_updates_health_and_invokes_launcher() -> None:
    topology = ClusterTopology(
        nodes=[
            NodeConfig(name="trainer-a", host="192.168.0.10"),
            NodeConfig(name="trainer-b", host="192.168.0.11"),
        ]
    )
    orchestrator = EnterpriseOrchestrator(topology, health_timeout=5.0)
    job = TrainingJob(name="demo", entrypoint="python", args=("-m", "train"))
    launches: list[tuple[str, int]] = []

    def _launcher(spec: LaunchSpec) -> None:
        launches.append((spec.node.name, spec.rank))

    plan = orchestrator.schedule(job, launcher=_launcher, dry_run=False)
    assert launches == [(spec.node.name, spec.rank) for spec in plan]
    report = orchestrator.health_report()
    for node in topology.trainers():
        assert report[node.name]["status"] == "healthy"


def test_default_launcher_executes_commands(monkeypatch) -> None:
    topology = ClusterTopology(nodes=[NodeConfig(name="trainer", host="127.0.0.1")])
    orchestrator = EnterpriseOrchestrator(topology)
    job = TrainingJob(name="demo", entrypoint="python", args=("-c", "print('ok')"))
    launched: list[dict[str, object]] = []

    class _Proc:
        def __init__(self, command, **kwargs):
            launched.append({"command": command, "env": kwargs.get("env", {})})
            self.args = command

        def wait(self) -> int:
            return 0

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            launched.append({"terminated": True})

    monkeypatch.setattr("entropy_news.model.orchestration.subprocess.Popen", _Proc)
    plan = orchestrator.schedule(job, dry_run=False)
    assert plan
    orchestrator.wait_for_processes()
    assert launched
    record = launched[0]
    assert record["command"] == ["python", "-c", "print('ok')"]
    env = record["env"]
    assert env["RANK"] == "0"
    assert env["WORLD_SIZE"] == "1"
    assert env["MASTER_ADDR"] == "127.0.0.1"


def test_cli_launch_triggers_execution(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def _load_topology(_path):
        return ClusterTopology(nodes=[NodeConfig(name="trainer", host="127.0.0.1")])

    def _schedule(self, job, launcher=None, *, dry_run=True):
        captured["dry_run"] = dry_run
        captured["launcher_none"] = launcher is None
        return [LaunchSpec(
            node=NodeConfig(name="trainer", host="127.0.0.1"),
            rank=0,
            local_rank=0,
            world_size=1,
            command=list(job.command()),
            env={"RANK": "0"},
        )]

    def _wait(self, *, check=True):
        captured["wait_called"] = True
        captured["check"] = check

    monkeypatch.setattr("entropy_news.model.orchestration._load_topology", _load_topology)
    monkeypatch.setattr(EnterpriseOrchestrator, "schedule", _schedule)
    monkeypatch.setattr(EnterpriseOrchestrator, "wait_for_processes", _wait)
    monkeypatch.setattr(sys, "argv", ["entropy-news-orchestrate", "--launch"])
    main()
    output = capsys.readouterr().out
    assert "RANK" in output
    assert captured.get("dry_run") is False
    assert captured.get("launcher_none") is True
    assert captured.get("wait_called") is True


def test_schedule_records_plan_failure(monkeypatch) -> None:
    topology = ClusterTopology(nodes=[NodeConfig(name="trainer", host="127.0.0.1")])
    orchestrator = EnterpriseOrchestrator(topology)
    job = TrainingJob(name="boom", entrypoint="python")

    def _boom(self, _job):
        raise RuntimeError("launch plan failed")

    monkeypatch.setattr(EnterpriseOrchestrator, "build_launch_plan", _boom)
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
