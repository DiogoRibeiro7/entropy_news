"""Reporting helpers for causal inference outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from .models import (
    DifferenceInDifferencesResult,
    SyntheticControlResult,
    TwoStageLeastSquaresResult,
)


@dataclass(frozen=True)
class PolicyScenario:
    """Description of a counterfactual policy scenario."""

    name: str
    description: str
    target_group: str


def build_summary_table(
    did_result: DifferenceInDifferencesResult,
    *,
    iv_result: TwoStageLeastSquaresResult | None = None,
    sc_result: SyntheticControlResult | None = None,
) -> pd.DataFrame:
    """Create a consolidated summary table for causal estimates."""

    rows: list[Mapping[str, float | str]] = [
        {
            "estimator": "difference_in_differences",
            "effect": did_result.average_treatment_effect,
            "standard_error": did_result.standard_error,
            "ci_lower": did_result.confidence_interval[0],
            "ci_upper": did_result.confidence_interval[1],
        }
    ]
    if iv_result is not None:
        rows.append(
            {
                "estimator": "two_stage_least_squares",
                "effect": iv_result.coefficient,
                "standard_error": iv_result.standard_error,
                "ci_lower": iv_result.coefficient - 1.96 * iv_result.standard_error,
                "ci_upper": iv_result.coefficient + 1.96 * iv_result.standard_error,
            }
        )
    if sc_result is not None:
        rows.append(
            {
                "estimator": "synthetic_control",
                "effect": sc_result.effect_series.mean(),
                "standard_error": sc_result.effect_series.std(ddof=1),
                "ci_lower": sc_result.effect_series.quantile(0.025),
                "ci_upper": sc_result.effect_series.quantile(0.975),
            }
        )
    return pd.DataFrame(rows)


def format_policy_narrative(
    scenario: PolicyScenario,
    did_result: DifferenceInDifferencesResult,
) -> str:
    """Generate a short natural-language narrative for stakeholders."""

    effect = did_result.average_treatment_effect
    direction = "increase" if effect > 0 else "decrease"
    magnitude = abs(effect)
    return (
        f"Under the {scenario.name} scenario, targeting {scenario.target_group}, "
        f"we estimate a {direction} of {magnitude:.2f} units in the outcome. "
        "The confidence interval suggests this conclusion remains stable across "
        "our historical baselines."
    )


def prepare_counterfactual_series(
    sc_result: SyntheticControlResult,
    *,
    horizon: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Extract aligned actual and synthetic series for plotting."""

    series = pd.DataFrame(
        {
            "actual": sc_result.actual_series,
            "synthetic": sc_result.synthetic_series,
            "effect": sc_result.effect_series,
        }
    )
    if horizon is not None:
        index = list(horizon)
        series = series.loc[index]
    index_name = series.index.name or "index"
    return series.reset_index().rename(columns={index_name: "time"})
