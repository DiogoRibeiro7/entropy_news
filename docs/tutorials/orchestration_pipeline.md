# Orchestration Pipeline Tutorial

Learn how to schedule distributed training runs with the enterprise
orchestrator and monitor them across multi-node clusters.

## Overview

The {class}`entropy_news.model.orchestration.EnterpriseOrchestrator` coordinates
job lifecycle events. This tutorial walks through defining a topology, launching
jobs, and instrumenting health signals.

## 1. Define the Cluster Topology

Create a JSON file that describes your compute estate. The orchestrator CLI and
Python helpers understand the structure below.

```json
{
  "name": "research-cluster",
  "master_port": 29501,
  "shared_storage": "/mnt/checkpoints",
  "environment": {
    "OMP_NUM_THREADS": "8"
  },
  "nodes": [
    {"name": "trainer-0", "host": "10.0.0.10", "role": "trainer", "processes": 2},
    {"name": "trainer-1", "host": "10.0.0.11", "role": "trainer", "processes": 2},
    {"name": "monitor", "host": "10.0.0.50", "role": "monitor", "processes": 1}
  ]
}
```

Validate the topology with a short script that instantiates
{class}`entropy_news.model.orchestration.ClusterTopology` and calls
{meth}`ClusterTopology.validate`.

```bash
python - <<'PY'
import json
from pathlib import Path
from entropy_news.model.orchestration import ClusterTopology, NodeConfig

payload = json.loads(Path("configs/cluster.json").read_text())

topology = ClusterTopology(
    nodes=[NodeConfig(**node) for node in payload["nodes"]],
    master_port=payload.get("master_port", 29500),
    shared_storage=Path(payload["shared_storage"]) if payload.get("shared_storage") else None,
    checkpoint_subdir=payload.get("checkpoint_subdir", "checkpoints"),
    environment=payload.get("environment", {}),
)
topology.validate()
print(f"World size: {topology.world_size()}")
PY
```

## 2. Craft a Launch Specification

```python
from pathlib import Path
from entropy_news.model.orchestration import TrainingJob

spec = TrainingJob(
    name="nightly-train",
    entrypoint="entropy-news-train",
    args=(
        "--model-config", "configs/resolved.json",
        "--train-data", "s3://datasets/headlines.parquet",
        "--checkpoint", "/mnt/checkpoints/nightly.pt",
        "--epochs", "6",
    ),
    env={"CUDA_VISIBLE_DEVICES": "0,1"},
    checkpoint_dir=Path("/mnt/checkpoints"),
    max_retries=2,
)
```

Persist the job specification alongside the topology (for example in the same
configuration repository) so schedulers and reviewers can audit changes.

## 3. Submit and Monitor Jobs

```python
import json
from pathlib import Path
from entropy_news.model.orchestration import (
    ClusterTopology,
    EnterpriseOrchestrator,
    NodeConfig,
)
from entropy_news.utils.metrics import start_metrics_server

payload = json.loads(Path("configs/cluster.json").read_text())

topology = ClusterTopology(
    nodes=[NodeConfig(**node) for node in payload["nodes"]],
    master_port=payload.get("master_port", 29500),
    shared_storage=Path(payload["shared_storage"]) if payload.get("shared_storage") else None,
    checkpoint_subdir=payload.get("checkpoint_subdir", "checkpoints"),
    environment=payload.get("environment", {}),
)
orchestrator = EnterpriseOrchestrator(topology)

plan = orchestrator.schedule(spec, dry_run=True)
for launch in plan:
    print(f"Rank {launch.rank} -> {launch.node.name} :: {launch.command}")

# Execute the plan locally using the built-in launcher.
orchestrator.schedule(spec, dry_run=False)
orchestrator.wait_for_processes()

# Start the Prometheus exporter so monitoring/ dashboards can scrape metrics.
start_metrics_server(port=9100)

# Custom launchers are still supported when you need to fan out over SSH,
# Kubernetes Jobs, or cloud batch APIs.
def ssh_launcher(launch_spec):
    ...  # Replace with fabric/paramiko calls in your environment

orchestrator.schedule(spec, launcher=ssh_launcher, dry_run=False)
```

Monitor the resulting Prometheus metrics using the dashboards shipped in
`monitoring/grafana_dashboard.json`. The :doc:`../playbooks/incident_response`
guide explains how to respond to unhealthy signals.

## 4. Automate Regression Rehearsals

Reuse the rehearsal steps captured in :doc:`../multi_node_rehearsal` and
:doc:`../playbooks/enterprise_rollout` to validate new releases before rolling
out to production clusters.
