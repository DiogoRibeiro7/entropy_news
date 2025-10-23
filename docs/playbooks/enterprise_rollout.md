# Enterprise Rollout Playbook

Coordinate the release of a new Entropy News model across enterprise
infrastructure while maintaining observability and rollback coverage.

## Phase 0 – Readiness Review

- Confirm completion of the :doc:`../tutorials/orchestration_pipeline` and
  :doc:`../tutorials/inference_delivery` tutorials for the release candidate.
- Validate documentation currency using the checklists in
  :doc:`../documentation_overhaul_plan`.
- Gather approvals from research, operations, and security stakeholders.

## Phase 1 – Preproduction Rehearsal

1. Provision the rehearsal environment using the topology in
   {mod}`entropy_news.model.orchestration.ClusterTopology`.
2. Deploy the training and inference stacks via the runbook in
   :doc:`../runbooks/enterprise_training`.
3. Execute the multi-node rehearsal from :doc:`../multi_node_rehearsal` and log
   deviations in the rollout tracker.

## Phase 2 – Production Launch

1. Schedule the initial training job window with the orchestrator and confirm
   Prometheus scrapes update in `monitoring/prometheus.yml`.
2. Promote the inference artefacts generated in the rehearsal to production
   storage and update downstream consumers.
3. Announce the launch window to stakeholders and distribute the monitoring
   dashboards in `monitoring/grafana_dashboard.json`.

## Phase 3 – Hypercare and Rollback

- Track latency, throughput, and entropy drift metrics hourly for the first
  48 hours. Capture anomalies in the :doc:`incident_response` logbook.
- If rollback is required, run the `rollback` scenario defined in
  :doc:`../runbooks/enterprise_training` and validate the canary dataset with
  `entropy-news-forecast --model-path output/checkpoints/latest.pt --new-data data/canary.csv`.
- Close the rollout by publishing a summary report that links to the causal
  analysis outputs (see :doc:`../tutorials/forecasting_insights`).
