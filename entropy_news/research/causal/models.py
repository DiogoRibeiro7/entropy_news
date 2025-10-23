"""Reusable causal estimators for Entropy News research experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .data import CausalPanelConfig


@dataclass(frozen=True)
class DifferenceInDifferencesResult:
    """Summary of a Difference-in-Differences estimate."""

    average_treatment_effect: float
    standard_error: float
    confidence_interval: tuple[float, float]
    treated_pre_mean: float
    treated_post_mean: float
    control_pre_mean: float
    control_post_mean: float


@dataclass(frozen=True)
class TwoStageLeastSquaresResult:
    """Outcome of a two-stage least squares estimation."""

    coefficient: float
    standard_error: float
    first_stage_f_stat: float
    residual_variance: float


@dataclass(frozen=True)
class SyntheticControlResult:
    """Synthetic control comparison for a treated unit."""

    treated_unit: str
    donor_units: Sequence[str]
    weights: pd.Series
    actual_series: pd.Series
    synthetic_series: pd.Series
    effect_series: pd.Series


def _validate_columns(panel: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns).difference(panel.columns)
    if missing:
        raise KeyError(f"Panel missing required columns: {sorted(missing)}")


def difference_in_differences(
    panel: pd.DataFrame,
    config: CausalPanelConfig,
) -> DifferenceInDifferencesResult:
    """Compute a classical difference-in-differences estimator."""

    _validate_columns(
        panel,
        [
            config.treatment_col,
            config.post_treatment_col,
            config.outcome_col,
        ],
    )
    treated = panel[panel[config.treatment_col] == 1]
    control = panel[panel[config.treatment_col] == 0]
    if treated.empty or control.empty:
        raise ValueError("Both treated and control groups are required for DiD")

    def _mean(group: pd.DataFrame, post: bool) -> float:
        mask = group[config.post_treatment_col] == (1 if post else 0)
        if not mask.any():
            raise ValueError("Missing pre/post observations for DiD computation")
        return float(group.loc[mask, config.outcome_col].mean())

    treated_pre = _mean(treated, post=False)
    treated_post = _mean(treated, post=True)
    control_pre = _mean(control, post=False)
    control_post = _mean(control, post=True)

    att = (treated_post - treated_pre) - (control_post - control_pre)

    def _variance(group: pd.DataFrame, post: bool) -> float:
        mask = group[config.post_treatment_col] == (1 if post else 0)
        variance = group.loc[mask, config.outcome_col].var(ddof=1)
        return float(variance) if pd.notna(variance) else 0.0

    treated_pre_var = _variance(treated, post=False)
    treated_post_var = _variance(treated, post=True)
    control_pre_var = _variance(control, post=False)
    control_post_var = _variance(control, post=True)

    treated_pre_n = int((treated[config.post_treatment_col] == 0).sum())
    treated_post_n = int((treated[config.post_treatment_col] == 1).sum())
    control_pre_n = int((control[config.post_treatment_col] == 0).sum())
    control_post_n = int((control[config.post_treatment_col] == 1).sum())

    for group_n in (treated_pre_n, treated_post_n, control_pre_n, control_post_n):
        if group_n == 0:
            raise ValueError("Insufficient observations for variance estimation")
    variance = (
        treated_post_var / treated_post_n
        + treated_pre_var / treated_pre_n
        + control_post_var / control_post_n
        + control_pre_var / control_pre_n
    )
    se = float(np.sqrt(variance))
    z = 1.96
    ci = (att - z * se, att + z * se)
    return DifferenceInDifferencesResult(
        average_treatment_effect=float(att),
        standard_error=se,
        confidence_interval=ci,
        treated_pre_mean=treated_pre,
        treated_post_mean=treated_post,
        control_pre_mean=control_pre,
        control_post_mean=control_post,
    )


def two_stage_least_squares(
    panel: pd.DataFrame,
    config: CausalPanelConfig,
    *,
    covariate_cols: Sequence[str] | None = None,
) -> TwoStageLeastSquaresResult:
    """Run a simplified two-stage least squares estimator."""

    if not config.instrument_cols:
        raise ValueError("At least one instrument is required for 2SLS")
    instrument_cols = list(config.instrument_cols)
    covariate_cols = list(covariate_cols or config.covariate_cols or [])
    _validate_columns(
        panel,
        [config.treatment_col, config.outcome_col, *instrument_cols, *covariate_cols],
    )

    def _design_matrix(columns: Sequence[str]) -> np.ndarray:
        return np.column_stack([
            np.ones(len(panel)),
            *[panel[col].to_numpy() for col in columns],
        ])

    # First stage regression of treatment on instruments and covariates
    z_matrix = _design_matrix(instrument_cols + covariate_cols)
    treatment = panel[config.treatment_col].to_numpy()
    first_stage_coef, _, _, _ = np.linalg.lstsq(z_matrix, treatment, rcond=None)
    treatment_hat = z_matrix @ first_stage_coef
    residual = treatment - treatment_hat
    ssr_unrestricted = float(residual.T @ residual)
    n_obs = len(treatment)
    k_instruments = len(instrument_cols)
    k_total = z_matrix.shape[1] - 1
    treatment_centered = treatment - treatment.mean()
    ss_total = float(treatment_centered.T @ treatment_centered)
    ssr_restricted = max(ss_total - ssr_unrestricted, 0.0)
    denominator = max(n_obs - k_total - 1, 1)
    numerator = max(k_instruments, 1)
    if ssr_unrestricted <= 0:
        f_stat = 0.0
        residual_variance = 0.0
    else:
        f_stat = float(((ssr_restricted / numerator) / (ssr_unrestricted / denominator)))
        residual_variance = ssr_unrestricted / max(denominator, 1)

    # Second stage using predicted treatment and covariates
    x_matrix = np.column_stack([np.ones(len(panel)), treatment_hat, *[panel[col].to_numpy() for col in covariate_cols]])
    outcome = panel[config.outcome_col].to_numpy()
    beta, _, _, _ = np.linalg.lstsq(x_matrix, outcome, rcond=None)
    fitted = x_matrix @ beta
    errors = outcome - fitted
    sigma2 = float((errors @ errors) / max(len(outcome) - x_matrix.shape[1], 1))
    xtx_inv = np.linalg.inv(x_matrix.T @ x_matrix)
    se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    return TwoStageLeastSquaresResult(
        coefficient=float(beta[1]),
        standard_error=se,
        first_stage_f_stat=f_stat,
        residual_variance=residual_variance,
    )


def synthetic_control(
    panel: pd.DataFrame,
    config: CausalPanelConfig,
    *,
    treated_unit: str,
    donor_units: Sequence[str],
) -> SyntheticControlResult:
    """Construct a synthetic control from donor units."""

    _validate_columns(panel, [config.unit_col, config.time_col, config.outcome_col])
    donor_units = list(donor_units)
    if treated_unit in donor_units:
        raise ValueError("Treated unit cannot appear in donor set")

    treated_data = panel[panel[config.unit_col] == treated_unit]
    donor_data = panel[panel[config.unit_col].isin(donor_units)]
    if treated_data.empty:
        raise ValueError("No observations for treated unit")
    if donor_data.empty:
        raise ValueError("No donor units provided")

    pivot_donors = donor_data.pivot_table(
        index=config.time_col,
        columns=config.unit_col,
        values=config.outcome_col,
    ).sort_index()
    treated_series = (
        treated_data.set_index(config.time_col)[config.outcome_col]
        .reindex(pivot_donors.index)
        .interpolate()
    )
    donor_matrix = pivot_donors.to_numpy()
    treated_vector = treated_series.to_numpy()
    weights, _, _, _ = np.linalg.lstsq(donor_matrix, treated_vector, rcond=None)
    weights = np.clip(weights, 0.0, None)
    if weights.sum() == 0:
        weights = np.full_like(weights, fill_value=1.0 / len(weights))
    else:
        weights = weights / weights.sum()
    synthetic_values = pivot_donors @ weights
    effect_values = treated_series.to_numpy() - synthetic_values
    weight_series = pd.Series(weights, index=pivot_donors.columns, name="weight")
    synthetic_series = pd.Series(synthetic_values, index=pivot_donors.index, name="synthetic")
    effect_series = pd.Series(effect_values, index=pivot_donors.index, name="effect")
    return SyntheticControlResult(
        treated_unit=treated_unit,
        donor_units=tuple(donor_units),
        weights=weight_series,
        actual_series=treated_series.rename("actual"),
        synthetic_series=synthetic_series,
        effect_series=effect_series,
    )
