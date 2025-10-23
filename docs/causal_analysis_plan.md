# Causal Analysis Module Design

The Release 1.0 roadmap introduces a causal analysis workstream to explain how
entropy-driven signals influence downstream market behaviour. This document
captures the early design so research stakeholders can iterate on methodology
in parallel with implementation.

## Objectives

1. **Quantify causal relationships** between entropy measures and market
   outcomes such as returns, volatility, or liquidity shocks.
2. **Support counterfactual reasoning** so analysts can estimate the impact of
   hypothetical news flows or policy interventions.
3. **Deliver reproducible artefacts**—code modules, datasets, and documentation
   that align with Entropy News quality standards.

## Collaboration Framework

| Track | Participants | Outcomes |
| --- | --- | --- |
| Methodology | Research partners (Diogo Ribeiro, external academics) | Identification strategies, assumptions, validation criteria |
| Engineering | Entropy News maintainers | Reusable causal estimators, dataset pipelines, evaluation harness |
| Product | Applied users & operations | Decision workflows, reporting requirements, monitoring hooks |

Monthly syncs will review progress, unblock data access questions, and align on
acceptance criteria for the Release 1.0 milestone.

## Deliverables

### Documentation

- `docs/causal_analysis_plan.md` (this file) tracks scope, milestones, and
  methodology references.
- A detailed methodological appendix summarising identification techniques,
  assumptions, and diagnostic tests.
- Integration notes for the deployment runbooks outlining how causal insights
  surface in dashboards and operator workflows.

### Notebooks

- `notebooks/causal_analysis_outline.ipynb` documents the canonical flow for data preparation, model fitting, and sensitivity checks. Researchers can fork the notebook to prototype alternatives without blocking library development.
- `notebooks/causal_counterfactual_playbook.ipynb` provides a runnable baseline showing how to parameterise scenarios and export counterfactual series for dashboards.
- Future notebooks will extend the outline with real datasets, parameter
  sweeps, placebo tests, and reporting utilities.

### Code Modules

Planned modules (naming subject to review with stakeholders):

- `entropy_news/research/causal/data.py` – dataset assembly helpers for
  propensity and instrumental-variable pipelines.
- `entropy_news/research/causal/models.py` – reusable estimators (DID, IV,
  synthetic control) with shared diagnostics.
- `entropy_news/research/causal/reporting.py` – summary tables, uplift plots,
  and policy narratives.

These modules will ship once the notebook prototypes stabilise.


## Implementation Status (May 2026 Update)

- The research toolkit now ships as `entropy_news.research.causal` with data assembly, estimators, and reporting helpers.
- A validated synthetic dataset and pytest harness exercise propensity features, DiD, 2SLS, and synthetic control flows.
- Baseline notebooks demonstrate end-to-end usage and link to the methodological appendix.

## Methodological Roadmap

1. **Exploratory Analysis (Q4 2025)**
   - Profile entropy and market variables to identify candidate covariates.
   - Run preliminary Granger causality and correlation studies for signal
     prioritisation.
2. **Model Design (Q1 2026)**
   - Select identification strategies for short- and long-horizon effects.
   - Define data windows, control variables, and instrument availability.
   - Document assumptions and falsification tests in the methodological
     appendix.
3. **Prototype Implementation (Q1-Q2 2026)**
   - Implement baseline DID and IV estimators with notebook demonstrations.
   - Add sensitivity analyses (placebo windows, covariate balance, bounding
     exercises).
4. **Operationalisation (Q2 2026)**
   - Integrate causal outputs into dashboards and deployment guides.
   - Automate report generation for recurring research reviews.
   - Capture monitoring hooks (data freshness, estimator drift) for Release 1.0
     observability.

## Dependencies & Open Questions

- **Data access:** confirm licensing for historical market data required for
  causal instruments and controls.
- **Compute budget:** estimate resources for repeated counterfactual
  simulations, especially under multi-node training workflows.
- **Evaluation metrics:** agree on success criteria (e.g., uplift magnitude,
  confidence intervals) with stakeholders to drive acceptance tests.

Feedback on this plan should be logged via Git issues tagged `causal-analysis`
so updates can be prioritised alongside engineering work.
