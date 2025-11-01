# Research Registry Playbook

Maintain a durable audit trail for experiments, causal studies, and production
rollouts.

## Registry Structure

- **Studies** – Link to notebooks (see `notebooks/`), datasets, and API
  references.
- **Models** – Capture configuration hashes from
  {mod}`entropy_news.model.config.ModelConfig` and checkpoint URIs.
- **Decisions** – Record approvals, rollbacks, and incident retrospectives.

Implement the registry as a version-controlled directory or database table with
the following minimal schema:

| Field | Description |
| ----- | ----------- |
| `id` | Unique identifier for the entry |
| `type` | `study`, `model`, or `decision` |
| `submitted_by` | Owner from research, engineering, or operations |
| `artifacts` | Paths to notebooks, reports, or datasets |
| `status` | `draft`, `approved`, `deprecated`, or `rolled_back` |
| `links` | References to tutorials, playbooks, and runbooks |

## Operating the Registry

1. **Ingest** – Use {func}`entropy_news.utils.cli.register_artifact` (planned)
   or submit a pull request that adds metadata files to the registry directory.
2. **Review** – Apply the :doc:`causal_review` and :doc:`enterprise_rollout`
   checklists during approval.
3. **Publish** – Announce new entries in release notes and embed the media assets
   recommended in :doc:`../media/storyboard`.

## Governance

- Enforce schema validation with `pytest tests/test_registry_contract.py` (add to
  CI) to avoid drift.
- Archive superseded artefacts in cold storage after the retention period.
- Align terminology with the API reference (:doc:`../api/index`) and tutorials so
  contributors can navigate seamlessly across documentation surfaces.
