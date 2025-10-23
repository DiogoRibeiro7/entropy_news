# Dashboard Story Tutorial

Create an interactive analytics narrative using the Streamlit dashboard to
highlight entropy movements for stakeholders.

## Objectives

1. Deploy the dashboard locally or via Docker Compose.
2. Curate a storyline that explains recent entropy spikes.
3. Export annotated artefacts for executive briefings.

## Launch the Dashboard

```bash
streamlit run entropy_news/dashboard/app.py -- --results output/rolling_forecast_results.csv
```

When running inside Docker Compose, enable the new `dashboard` profile defined
in :doc:`../deployment`:

```bash
docker compose --profile dashboard up
```

## Craft the Narrative

- Use the metric selectors to focus on entropy dimensions relevant to the
  audience (market-wide vs. sector-specific).
- Apply {func}`entropy_news.dashboard.app.filter_by_month` to isolate periods of
  volatility and annotate them with policy or macro events.
- Generate textual summaries with
  {func}`entropy_news.dashboard.app.build_report` and cross-check them with the
  causal insights produced in :doc:`forecasting_insights`.

## Export Artefacts

- Capture charts as PNG using the built-in Streamlit export controls.
- Convert markdown summaries to presentation-ready PDFs with tools such as
  `pandoc` or `nbconvert`, bundling screenshots, commentary, and causal
  diagnostics captured from the research tooling.
- Follow the :doc:`../media/storyboard` checklist to script a video walkthrough
  if stakeholders prefer asynchronous updates.
