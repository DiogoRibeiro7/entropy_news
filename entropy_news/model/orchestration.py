"""Enterprise orchestration layer for multi-node training workflows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Sequence

from entropy_news.utils.metrics import (
    observe_job_duration,
    record_heartbeat_age,
    record_launch_plan_failure,
    record_launch_plan_size,
    record_rank_launch,
    start_metrics_server,
    update_active_processes,
)

from .distributed import CheckpointManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NodeConfig:
    """Describe a compute node participating in an orchestrated job."""

    name: str
    host: str
    role: str = "trainer"
    gpus: int = 0
    processes: int = 1
    tags: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Ensure the configuration contains sensible values."""

        if not self.name:
            raise ValueError("NodeConfig.name must be provided")
        if not self.host:
            raise ValueError("NodeConfig.host must be provided")
        if self.processes < 1:
            raise ValueError("NodeConfig.processes must be at least 1")
        if self.gpus < 0:
            raise ValueError("NodeConfig.gpus cannot be negative")
        if self.role not in {"trainer", "evaluator", "monitor"}:
            raise ValueError(f"Unsupported role: {self.role}")


@dataclass(slots=True)
class ClusterTopology:
    """A declarative view of nodes and orchestration defaults."""

    nodes: List[NodeConfig]
    master_port: int = 29500
    shared_storage: Path | None = None
    checkpoint_subdir: str = "checkpoints"
    environment: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the topology before executing jobs."""

        if self.master_port <= 0:
            raise ValueError("master_port must be positive")
        seen = set()
        trainers = 0
        for node in self.nodes:
            node.validate()
            if node.name in seen:
                raise ValueError(f"Duplicate node name detected: {node.name}")
            seen.add(node.name)
            if node.role == "trainer":
                trainers += 1
        if trainers == 0:
            raise ValueError("At least one trainer node is required")

    def world_size(self) -> int:
        """Return the total number of processes to be launched."""

        return sum(node.processes for node in self.nodes if node.role == "trainer")

    def trainers(self) -> List[NodeConfig]:
        """Return trainer nodes ordered as defined in the configuration."""

        return [node for node in self.nodes if node.role == "trainer"]

    def checkpoint_directory(self) -> Path | None:
        """Resolve the checkpoint directory for orchestrated jobs."""

        if self.shared_storage is None:
            return None
        return self.shared_storage / self.checkpoint_subdir


@dataclass(slots=True)
class TrainingJob:
    """Describe an orchestrated training job."""

    name: str
    entrypoint: str
    args: Sequence[str] = field(default_factory=tuple)
    env: Dict[str, str] = field(default_factory=dict)
    checkpoint_dir: Path | None = None
    max_retries: int = 0

    def command(self) -> List[str]:
        """Return the entrypoint and arguments as a shell-safe list."""

        return [self.entrypoint, *self.args]


@dataclass(slots=True)
class LaunchSpec:
    """Computed launch specification for a single rank."""

    node: NodeConfig
    rank: int
    local_rank: int
    world_size: int
    command: List[str]
    env: Dict[str, str]


Launcher = Callable[["LaunchSpec"], subprocess.Popen[bytes] | None]


class EnterpriseOrchestrator:
    """Schedule and monitor enterprise-scale training jobs."""

    def __init__(self, topology: ClusterTopology, *, health_timeout: float = 60.0) -> None:
        topology.validate()
        self.topology = topology
        self.health_timeout = health_timeout
        self._heartbeats: Dict[str, float] = {}
        self._health_server: HTTPServer | None = None
        self._health_thread: threading.Thread | None = None
        self._running_processes: List[subprocess.Popen[bytes]] = []
        self._launch_start: float | None = None

    def build_launch_plan(self, job: TrainingJob) -> List[LaunchSpec]:
        """Construct launch specifications for each distributed rank."""

        world_size = self.topology.world_size()
        if world_size == 0:
            raise ValueError("No trainer processes defined in topology")
        trainers = self.topology.trainers()
        master_addr = trainers[0].host
        checkpoint_dir = job.checkpoint_dir or self.topology.checkpoint_directory()
        env_base = dict(self.topology.environment)
        env_base.update(job.env)
        if checkpoint_dir is not None:
            manager = CheckpointManager(checkpoint_dir)
            env_base.setdefault("ENTROPY_NEWS_CHECKPOINT_DIR", str(checkpoint_dir))
            env_base.setdefault("ENTROPY_NEWS_CHECKPOINT_MAX", str(manager.max_checkpoints))
        plan: List[LaunchSpec] = []
        global_rank = 0
        for node in trainers:
            for local_rank in range(node.processes):
                env = dict(env_base)
                env.update(
                    {
                        "MASTER_ADDR": master_addr,
                        "MASTER_PORT": str(self.topology.master_port),
                        "RANK": str(global_rank),
                        "LOCAL_RANK": str(local_rank),
                        "WORLD_SIZE": str(world_size),
                        "NODE_NAME": node.name,
                        "ROLE": node.role,
                    }
                )
                plan.append(
                    LaunchSpec(
                        node=node,
                        rank=global_rank,
                        local_rank=local_rank,
                        world_size=world_size,
                        command=job.command(),
                        env=env,
                    )
                )
                global_rank += 1
        return plan

    def schedule(
        self,
        job: TrainingJob,
        launcher: Launcher | None = None,
        *,
        dry_run: bool = True,
    ) -> List[LaunchSpec]:
        """Schedule ``job`` and optionally invoke ``launcher`` for each rank.

        When ``dry_run`` is ``False`` the orchestrator records a heartbeat for
        every launched rank so the health endpoint immediately reflects the
        deployment state.
        """

        try:
            plan = self.build_launch_plan(job)
        except Exception as exc:
            record_launch_plan_failure(type(exc).__name__)
            raise
        record_launch_plan_size(len(plan))
        self._running_processes = []
        if dry_run:
            update_active_processes(0)
            self._launch_start = None
            return plan
        if launcher is None:
            launcher = self.default_launcher
        self._launch_start = time.time()
        for spec in plan:
            self.register_heartbeat(spec.node.name)
            record_rank_launch(spec.node.name, spec.node.role)
            handle = launcher(spec)
            if handle is not None:
                self._running_processes.append(handle)
                update_active_processes(len(self._running_processes))
        return plan

    @staticmethod
    def default_launcher(spec: LaunchSpec) -> subprocess.Popen[bytes]:
        """Launch ``spec`` locally using ``subprocess.Popen``."""

        logger.info(
            "Launching rank %s on node %s with command %s", spec.rank, spec.node.name, spec.command
        )
        env = os.environ.copy()
        env.update(spec.env)
        return subprocess.Popen(spec.command, env=env)

    def wait_for_processes(self, *, check: bool = True) -> None:
        """Wait for launched processes to exit and optionally enforce success."""

        for process in list(self._running_processes):
            return_code = process.wait()
            if check and return_code != 0:
                raise RuntimeError(
                    f"Process {process.args!r} exited with status {return_code}"
                )
            self._running_processes.remove(process)
        update_active_processes(len(self._running_processes))
        if self._launch_start is not None:
            observe_job_duration(time.time() - self._launch_start)
            self._launch_start = None

    def terminate_processes(self) -> None:
        """Terminate any still-running launched processes."""

        for process in self._running_processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        self._running_processes.clear()
        update_active_processes(0)

    def register_heartbeat(self, node_name: str) -> None:
        """Record a liveness heartbeat for ``node_name``."""

        self._heartbeats[node_name] = time.time()
        record_heartbeat_age(node_name, 0.0, True)

    def health_report(self) -> Dict[str, Dict[str, float | str]]:
        """Return liveness information for all nodes."""

        now = time.time()
        status: Dict[str, Dict[str, float | str]] = {}
        for node in self.topology.nodes:
            heartbeat = self._heartbeats.get(node.name, 0.0)
            age = now - heartbeat if heartbeat else float("inf")
            is_healthy = age <= self.health_timeout
            status[node.name] = {
                "role": node.role,
                "host": node.host,
                "last_heartbeat": heartbeat,
                "latency": age,
                "status": "healthy" if is_healthy else "stale",
            }
            record_heartbeat_age(node.name, age, is_healthy)
        return status

    def start_health_server(self, host: str = "127.0.0.1", port: int = 0) -> int:
        """Expose health information via a lightweight HTTP endpoint."""

        if self._health_server is not None:
            raise RuntimeError("Health server already running")

        orchestrator = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # type: ignore[override]
                if self.path not in {"/", "/health"}:
                    self.send_response(404)
                    self.end_headers()
                    return
                report = orchestrator.health_report()
                payload = json.dumps(report).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:  # type: ignore[override]
                logger.debug("health_server: " + format, *args)

        server = HTTPServer((host, port), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._health_server = server
        self._health_thread = thread
        logger.info("Health server started on %s:%s", host, server.server_port)
        return int(server.server_port)

    def stop_health_server(self) -> None:
        """Stop the HTTP health endpoint if running."""

        if self._health_server is None:
            return
        self._health_server.shutdown()
        self._health_server.server_close()
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=1)
        self._health_server = None
        self._health_thread = None

    def iter_launches(self, plan: Iterable[LaunchSpec]) -> Iterator[LaunchSpec]:
        """Convenience generator that also records heartbeats as launches start."""

        for spec in plan:
            self.register_heartbeat(spec.node.name)
            yield spec


def _load_topology(path: Path | None) -> ClusterTopology:
    """Load a topology definition from ``path`` or fall back to localhost."""

    if path is None:
        return ClusterTopology(nodes=[NodeConfig(name="trainer", host="127.0.0.1")])
    content = json.loads(path.read_text(encoding="utf-8"))
    nodes = [
        NodeConfig(
            name=item["name"],
            host=item["host"],
            role=item.get("role", "trainer"),
            gpus=item.get("gpus", 0),
            processes=item.get("processes", 1),
            tags=item.get("tags", {}),
        )
        for item in content["nodes"]
    ]
    shared_storage = content.get("shared_storage")
    topology = ClusterTopology(
        nodes=nodes,
        master_port=content.get("master_port", 29500),
        shared_storage=Path(shared_storage) if shared_storage else None,
        checkpoint_subdir=content.get("checkpoint_subdir", "checkpoints"),
        environment=content.get("environment", {}),
    )
    return topology


def main() -> None:
    """Render launch plans or expose a health endpoint for the orchestrator."""

    parser = argparse.ArgumentParser(description="Enterprise orchestration helper")
    parser.add_argument("--topology", type=Path, default=None, help="Path to topology JSON file")
    parser.add_argument("--entrypoint", default="entropy-news-train", help="Training entrypoint command")
    parser.add_argument("--args", nargs=argparse.REMAINDER, help="Additional arguments for the entrypoint")
    parser.add_argument("--health-server", action="store_true", help="Start a health endpoint after rendering the plan")
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Execute the plan using the built-in launcher instead of performing a dry run",
    )
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        help="Expose Prometheus metrics for orchestrator scheduling telemetry.",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Optional port for the Prometheus metrics exporter (defaults to ENV or 8000).",
    )
    args = parser.parse_args()

    topology = _load_topology(args.topology)
    orchestrator = EnterpriseOrchestrator(topology)
    if args.enable_metrics:
        try:
            start_metrics_server(args.metrics_port)
        except OSError as exc:
            logger.warning("Failed to start Prometheus metrics server: %s", exc)
    job = TrainingJob(name="cli", entrypoint=args.entrypoint, args=tuple(args.args or ()))
    plan = orchestrator.schedule(job, dry_run=not args.launch)
    json_plan = [
        {
            "node": spec.node.name,
            "rank": spec.rank,
            "command": spec.command,
            "env": spec.env,
        }
        for spec in plan
    ]
    print(json.dumps(json_plan, indent=2))

    if args.launch:
        try:
            orchestrator.wait_for_processes()
        except KeyboardInterrupt:
            logger.info("Termination requested, stopping launched processes")
            orchestrator.terminate_processes()
            raise SystemExit(130)

    if args.health_server:
        port = orchestrator.start_health_server(host="0.0.0.0", port=9090)
        logger.info("Health server running on port %s", port)
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Stopping health server")
        finally:
            orchestrator.stop_health_server()


if __name__ == "__main__":
    main()
