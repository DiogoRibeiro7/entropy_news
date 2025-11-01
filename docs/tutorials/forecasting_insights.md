# Forecasting Insights Tutorial

Leverage the causal analysis toolkit to understand how specific events impact
entropy trajectories and downstream forecasts.

## Prerequisites

- Complete the :doc:`training_workflow` tutorial to produce a trained model and
  rolling forecasts.
- Prepare entropy and market datasets compatible with
  {class}`entropy_news.research.causal.data.CausalPanelConfig`.
- Install the optional research dependencies: `pip install -e .[research]`.

## 1. Assemble Panel Data

Use the dataset utilities to align outcomes and treatments across markets.

```python
from entropy_news.research.causal.data import (
    CausalPanelConfig,
    assemble_causal_panel,
    prepare_causal_dataset,
)
import pandas as pd

entropy_df = pd.read_parquet("data/entropy_forecasts.parquet")
market_df = pd.read_csv("data/policy_events.csv")
config = CausalPanelConfig(
    unit_col="ticker",
    time_col="timestamp",
    outcome_col="entropy_news",
    treatment_col="policy_flag",
    post_treatment_col="post_event",
    covariate_cols=("market_cap", "liquidity"),
    instrument_cols=("instrument_strength",),
)
panel = assemble_causal_panel(entropy_df, market_df, config)
dataset = prepare_causal_dataset(entropy_df, market_df, config)
```

The helpers standardise column names, impute gaps, and surface metadata so
downstream estimators can validate assumptions automatically.

## 2. Estimate Causal Effects

Apply the estimators individually to triangulate the expected effect size.

```python
from entropy_news.research.causal.models import (
    difference_in_differences,
    two_stage_least_squares,
    synthetic_control,
)

did_result = difference_in_differences(panel, config)
iv_result = two_stage_least_squares(panel, config)
sc_result = synthetic_control(
    panel,
    config,
    treated_unit="ACME",
    donor_units=["BETA", "GAMMA", "OMEGA"],
)
```

Inspect each result object to compare effect magnitudes, diagnostic statistics,
and donor weights.

## 3. Generate Reports

Summarise findings with the reporting helpers.

```python
from entropy_news.research.causal.reporting import (
    PolicyScenario,
    build_summary_table,
    format_policy_narrative,
    prepare_counterfactual_series,
)

summary = build_summary_table(did_result, iv_result=iv_result, sc_result=sc_result)
scenario = PolicyScenario(
    name="Targeted Stimulus",
    description="Assess entropy response to the targeted macro stimulus",
    target_group="large-cap equities",
)
narrative = format_policy_narrative(scenario, did_result)
series = prepare_counterfactual_series(sc_result)
summary.to_markdown("output/causal_summary.md")
series.to_csv("output/causal_series.csv", index=False)
with open("output/causal_narrative.txt", "w", encoding="utf-8") as fh:
    fh.write(narrative)
```

Embed the generated markdown, CSV, and narrative into research memos or dashboards.
Use the :doc:`../media/index` guidelines to pair the report with narrated video or
captioned screenshots for stakeholder reviews.

## 4. Operationalise the Workflow

- Schedule the pipeline via the enterprise orchestrator documented in
  :doc:`../playbooks/enterprise_rollout`.
- Validate that treatments satisfy assumptions with the diagnostic checklist in
  :doc:`../playbooks/causal_review`.
- Archive results and artefacts in the research registry described in
  :doc:`../playbooks/research_registry` to support reproducibility.
