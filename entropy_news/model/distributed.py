"""Distributed training helpers for enterprise deployments."""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator

try:  # pragma: no cover - optional torch dependency
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover - torch not installed
    torch = None  # type: ignore[assignment]
    dist = None

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Capture throughput statistics for distributed training loops."""

    scope: str
    start_time: float = field(default_factory=time.perf_counter)
    batches: int = 0
    samples: int = 0

    def record_batch(self, batch_size: int) -> None:
        """Record a processed batch of ``batch_size`` samples."""

        self.batches += 1
        self.samples += batch_size

    def summary(self) -> Dict[str, float]:
        """Return elapsed time and throughput metrics."""

        elapsed = time.perf_counter() - self.start_time
        throughput = (self.samples / elapsed) if elapsed > 0 else 0.0
        return {
            "elapsed": elapsed,
            "samples": float(self.samples),
            "batches": float(self.batches),
            "throughput": throughput,
        }


def init_distributed(backend: str = "nccl", init_method: str = "env://") -> None:
    """Initialise the torch distributed backend when available."""

    if dist is None:  # pragma: no cover - guard for non-torch environments
        raise RuntimeError("torch.distributed is required for distributed training")
    if dist.is_initialized():  # pragma: no cover - trivial branch
        return
    dist.init_process_group(backend=backend, init_method=init_method)
    logger.info("Distributed backend initialised with backend=%s", backend)


def synchronize_metrics(values: Iterable[float]) -> float:
    """Aggregate metrics across ranks by averaging ``values``."""

    values = list(values)
    if not values:
        raise ValueError("No values provided for synchronization")
    if dist is None or torch is None or not dist.is_initialized():  # pragma: no cover - simple branch
        return sum(values) / len(values)
    tensor = torch.tensor(values, dtype=torch.float32)
    dist.all_reduce(tensor)
    return float(tensor.mean().item())


@contextlib.contextmanager
def monitor_training(scope: str) -> Iterator[TrainingMetrics]:
    """Context manager that logs timing information for ``scope``."""

    metrics = TrainingMetrics(scope=scope)
    logger.info("Starting %s", scope)
    try:
        yield metrics
    finally:  # pragma: no branch - ensure logging happens even on error
        stats = metrics.summary()
        logger.info(
            "Completed %s in %.2fs (samples=%d, throughput=%.2f/s)",
            scope,
            stats["elapsed"],
            int(stats["samples"]),
            stats["throughput"],
        )


class CheckpointManager:
    """Persist and rotate checkpoints for distributed runs."""

    def __init__(
        self,
        directory: str | Path,
        *,
        max_checkpoints: int = 5,
        keep_every: int | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.keep_every = keep_every
        if dist is not None and dist.is_initialized():  # pragma: no branch - trivial
            self.rank = dist.get_rank()
        else:
            self.rank = 0

    def is_primary(self) -> bool:
        """Return ``True`` when the current rank should write checkpoints."""

        return self.rank == 0

    def save(self, state: Dict[str, Any], step: int) -> Path:
        """Persist ``state`` for ``step`` and prune older checkpoints."""

        if torch is None:  # pragma: no cover - requires torch in production
            raise RuntimeError("torch is required to save checkpoints")
        path = self.directory / f"checkpoint-{step}.pt"
        if not self.is_primary():
            return path
        torch.save(state, path)
        self._prune(step)
        return path

    def list_checkpoints(self) -> list[Path]:
        """Return checkpoints sorted from oldest to newest."""

        def sort_key(path: Path) -> tuple[int, float]:
            """Order by numeric step first, then modification time."""

            try:
                step = int(path.stem.split("-")[-1])
            except ValueError:
                step = 0
            return (step, path.stat().st_mtime)

        checkpoints = sorted(self.directory.glob("checkpoint-*.pt"), key=sort_key)
        return checkpoints

    def _prune(self, current_step: int) -> None:
        if self.keep_every and current_step % self.keep_every == 0:
            return
        checkpoints = self.list_checkpoints()
        while len(checkpoints) > self.max_checkpoints:
            candidate = checkpoints.pop(0)
            try:
                candidate.unlink()
            except FileNotFoundError:  # pragma: no cover - race condition guard
                continue


def stress_test(
    step: Callable[[], None],
    *,
    iterations: int = 10,
    synchronize: bool = True,
) -> Dict[str, float]:
    """Execute ``step`` ``iterations`` times and report timing statistics."""

    if synchronize and dist is not None and dist.is_initialized():  # pragma: no cover - requires torch dist
        dist.barrier()
    start = time.perf_counter()
    for _ in range(iterations):
        step()
    if synchronize and dist is not None and dist.is_initialized():  # pragma: no cover
        dist.barrier()
    duration = time.perf_counter() - start
    throughput = iterations / duration if duration > 0 else 0.0
    stats: Dict[str, float] = {
        "iterations": float(iterations),
        "duration": duration,
        "throughput": throughput,
    }
    if (
        dist is not None
        and torch is not None
        and dist.is_initialized()
        and dist.get_world_size() > 0
    ):  # pragma: no cover - requires torch
        tensor = torch.tensor([duration, throughput], dtype=torch.float32)
        dist.all_reduce(tensor)
        world_size = dist.get_world_size()
        stats["duration"] = float(tensor[0].item() / world_size)
        stats["throughput"] = float(tensor[1].item() / world_size)
    return stats
