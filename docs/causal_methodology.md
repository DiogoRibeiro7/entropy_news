# Causal Analysis Methodological Appendix

This appendix documents the statistical assumptions and diagnostics used by the
`entropy_news.research.causal` toolkit. It is intended to complement the
high-level roadmap by capturing the technical guardrails required for reliable
counterfactual analysis.

## Identification Strategies

### Difference-in-Differences (DiD)
- **Parallel trends**: Treated and control cohorts must evolve similarly absent
treatment. Analysts should review entropy trend plots and propensity diagnostics
from `build_propensity_features` to check alignment.
- **No anticipation**: News-driven interventions are assumed to impact the
outcome only after the `post_treatment_col` indicator activates. Abrupt pre-
trends should be investigated as potential assumption breaks.

### Two-Stage Least Squares (2SLS)
- **Instrument relevance**: The first-stage F-statistic reported in
`TwoStageLeastSquaresResult` should exceed the conventional threshold (≈10) to
avoid weak-instrument bias.
- **Exclusion restriction**: Instruments must influence market outcomes only
through the treatment channel. Logging the residual variance alongside domain
review of news coverage helps validate this assumption.

### Synthetic Control
- **Convex hull coverage**: Donor units should span the treated unit’s pre-
treatment outcome path. The toolkit clips and normalises weights, but analysts
should inspect residual effect series and adjust donor sets when the synthetic
fit is poor.

## Diagnostic Workflow

1. **Panel assembly** – Use `assemble_causal_panel` to merge entropy metrics with
   market outcomes and confirm schema compliance.
2. **Propensity diagnostics** – Review the rolling `outcome_trend` and
   `treatment_trend` series to assess balance and identify covariate drift.
3. **Estimator execution** – Run DiD, 2SLS, and synthetic control estimators via
the exposed helper functions and capture their result dataclasses for
reproducibility.
4. **Reporting** – Generate summary tables and narratives with
`build_summary_table` and `format_policy_narrative`. Counterfactual series can be
forwarded to dashboards using `prepare_counterfactual_series`.

## Reproducibility Checklist

- Version control the configuration (column mappings, donor sets, and horizons).
- Track the exact dataset hashes for entropy metrics and market outcomes.
- Export generated tables and narratives to the documentation portal for audit
trails.
