# Enterprise Training Runbook

This runbook documents the operational steps for launching and maintaining
multi-node training workloads with the Entropy News orchestration layer. It
covers bare-metal deployments, Kubernetes clusters, and cloud-managed training
services so enterprise operators can standardise procedures across environments.

## 1. Preparation Checklist

1. **Cluster configuration:** Create a `ClusterTopology` document that lists all
   trainer, evaluator, and monitoring nodes. Store the JSON or YAML artefact in
   your configuration repository and validate it with
   `ClusterTopology(nodes=...).validate()`.
2. **Credentials:** Ensure access to container registry, shared storage, and log
   aggregation backends. Rotate secrets prior to major runs.
3. **Artifacts:** Build the training container image and push it to the target
   registry. Confirm that the `entropy-news-train` CLI is present.
4. **Monitoring:** Deploy the Prometheus and Grafana assets from the
   `monitoring/` directory and confirm that scrape jobs succeed.

## 2. Bare-metal Deployment

1. **Provisioning:** Install Docker and NVIDIA drivers on each node. Mount the
   shared filesystem to the same path across machines (for example `/mnt/entropy`).
2. **Configuration:** Copy the orchestration configuration to `/etc/entropy-news/`.
   The snippet below launches a three-node job:

   ```python
   from entropy_news.model.orchestration import ClusterTopology, EnterpriseOrchestrator, NodeConfig, TrainingJob

   topology = ClusterTopology(
       nodes=[
           NodeConfig(name="trainer-a", host="trainer-a", processes=2),
           NodeConfig(name="trainer-b", host="trainer-b", processes=2),
           NodeConfig(name="monitor", host="monitor", role="monitor"),
       ],
       shared_storage=Path("/mnt/entropy"),
   )
   orchestrator = EnterpriseOrchestrator(topology)
   job = TrainingJob(name="release-candidate", entrypoint="entropy-news-train")
   orchestrator.schedule(job, dry_run=False)
   ```

3. **Execution:** Use SSH or an MPS scheduler to invoke the launch plan on each
   host. The plan exposes `MASTER_ADDR`, `WORLD_SIZE`, and rank information in the
   environment, so `torchrun` is not required unless desired.
4. **Recovery:** If a node fails, drain it via the hardware console, mark the
   node as `role="monitor"`, and rerun `schedule()` to compute a new plan. Resume
   training from the latest checkpoint recorded in the shared directory.

## 3. Kubernetes Deployment

1. **Namespaces:** Create a dedicated namespace `entropy-news-enterprise`.
2. **Helm values:** Use the sample Helm overrides in `docs/runbooks/helm-values.yaml`
   (create per environment) to define node pools and storage classes.
3. **Launcher:** The orchestrator emits per-rank environment variables. Map each
   `LaunchSpec` to a Kubernetes Job template with init containers that fetch the
   training configuration and secrets.
4. **Scaling:** Adjust the `processes` field per node to match GPU counts. The
   `world_size` automatically scales based on the topology definition.
5. **Rollback:** Use Kubernetes Job history and the runbook's metrics dashboard
   to identify failing pods. Delete unhealthy pods and let the Job controller
   reschedule them; the checkpoint manager resumes progress.

## 4. Cloud-managed Training Services

1. **Cloud profiles:** For services such as AWS SageMaker or Azure ML, map each
   node to the provider's worker definition. Export the `LaunchSpec` data to JSON
   and supply it as part of the job configuration.
2. **Security:** Apply IAM roles or service principals with least-privilege
   policies to access storage, container registries, and monitoring endpoints.
3. **Autoscaling:** Configure managed services to respect the orchestrator's
   `WORLD_SIZE` and `RANK` assignments. Do not allow the service to add workers on
   the fly without recalculating the launch plan.
4. **Audit:** Store the rendered launch plan with deployment metadata so audit
   trails capture the command lines and environment variables used for each run.

## 5. Incident Response

1. **Node heartbeat loss:** Check the health endpoint (default `/health`). If the
   node is `stale`, attempt to restart the process. If unsuccessful, mark the node
   offline and recompute the plan.
2. **Checkpoint issues:** Inspect the shared storage path. The `CheckpointManager`
   rotates files automatically; increase `max_checkpoints` in the orchestrator
   configuration if retention is too aggressive.
3. **Performance regressions:** Review the Prometheus metrics for throughput and
   gradient health. Compare against the baseline Grafana dashboard to spot
   anomalies.

## 6. Post-run Activities

1. Archive checkpoints and logs to long-term storage.
2. Update the release documentation with metrics captured during the run.
3. Capture lessons learned and feed them back into the cluster configuration for
   the next rehearsal.

For quick references and templates, see the accompanying files in this directory
and the monitoring assets described in `monitoring/README.md`.
