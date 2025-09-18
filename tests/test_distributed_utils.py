from __future__ import annotations

from pathlib import Path

import pytest

from entropy_news.model.distributed import (
    CheckpointManager,
    init_distributed,
    monitor_training,
    stress_test,
    synchronize_metrics,
)


def test_synchronize_metrics_without_dist() -> None:
    result = synchronize_metrics([1.0, 3.0, 5.0])
    assert result == pytest.approx(3.0)


def test_monitor_training_logs(caplog) -> None:
    with caplog.at_level("INFO"):
        with monitor_training("unit-test") as metrics:
            metrics.record_batch(5)
    messages = [record.message for record in caplog.records]
    assert "Starting unit-test" in messages[0]
    assert "Completed unit-test" in messages[-1]


def test_init_distributed_raises_when_missing(monkeypatch) -> None:
    import types

    fake_dist = types.SimpleNamespace(is_initialized=lambda: False)
    monkeypatch.setattr("entropy_news.model.distributed.dist", None)
    with pytest.raises(RuntimeError):
        init_distributed()
    monkeypatch.setattr("entropy_news.model.distributed.dist", fake_dist)
    with pytest.raises(AttributeError):  # missing init_process_group
        init_distributed()


def test_checkpoint_manager_rotates(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    from entropy_news.model.distributed import torch as torch_mod

    manager = CheckpointManager(tmp_path, max_checkpoints=2)

    def fake_save(state, path):  # type: ignore[override]
        Path(path).write_text(str(state))

    monkeypatch.setattr(torch_mod, "save", fake_save)

    manager.save({"epoch": 1}, step=1)
    manager.save({"epoch": 2}, step=2)
    manager.save({"epoch": 3}, step=3)

    checkpoints = manager.list_checkpoints()
    assert len(checkpoints) <= 2
    assert any("checkpoint-3" in str(cp) for cp in checkpoints)


def test_stress_test_reports_metrics() -> None:
    calls = {"count": 0}

    def step() -> None:
        calls["count"] += 1

    stats = stress_test(step, iterations=5, synchronize=False)
    assert stats["iterations"] == pytest.approx(5.0)
    assert stats["throughput"] >= 0.0
    assert calls["count"] == 5
