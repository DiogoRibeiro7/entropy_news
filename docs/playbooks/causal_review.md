# Causal Review Playbook

Use this playbook to vet causal analysis studies before they inform product or
policy decisions.

## Intake

- Capture the study hypothesis, identification strategy, and dataset inventory
  using the template in :doc:`../causal_analysis_plan`.
- Ensure datasets comply with governance requirements by cross-referencing
  :doc:`../data_formats`.

## Validation Checklist

1. **Assumptions** – For each estimator in
   {mod}`entropy_news.research.causal.models`, confirm that the necessary
   assumptions (parallel trends, exclusion restrictions, stable unit treatment)
   hold. Document supporting evidence or caveats.
2. **Diagnostics** – Review effect plots generated via
   {func}`entropy_news.research.causal.reporting.render_time_series` and check
   residual analyses for anomalies.
3. **Robustness** – Compare
   {func}`entropy_news.research.causal.models.two_stage_least_squares` and
   {func}`entropy_news.research.causal.models.synthetic_control` outputs against
   the baseline difference-in-differences estimate, documenting any divergence in
   the review log.

## Approval

- Compile a summary memo with {func}`entropy_news.research.causal.reporting.build_summary`.
- Present findings in a multimedia format (see :doc:`../media/index`) so
  decision-makers can consume results asynchronously.
- Record approval decisions and follow-up actions in the research registry
  documented in :doc:`research_registry`.

## Post-Deployment Monitoring

- Track live entropy deltas against the counterfactual estimates and alert the
  incident team if variance exceeds agreed thresholds.
- Schedule quarterly refreshes of the analysis as new data becomes available.
- Update :doc:`../documentation_overhaul_plan` with lessons learned to improve
  future tutorials and API references.
