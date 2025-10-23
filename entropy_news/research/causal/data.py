"""Dataset assembly helpers for causal inference experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class CausalPanelConfig:
    """Column mapping for causal panel datasets.

    Attributes:
        unit_col: Column identifying observational units.
        time_col: Column representing temporal ordering.
        outcome_col: Target outcome used in causal estimators.
        treatment_col: Indicator for treated observations.
        post_treatment_col: Indicator for post-treatment period.
        covariate_cols: Optional baseline covariates to retain.
        instrument_cols: Optional names of instrumental variables.
    """

    unit_col: str
    time_col: str
    outcome_col: str
    treatment_col: str
    post_treatment_col: str
    covariate_cols: Sequence[str] | None = None
    instrument_cols: Sequence[str] | None = None


@dataclass(frozen=True)
class CausalDataset:
    """Container object bundling prepared causal analysis assets."""

    panel: pd.DataFrame
    propensity_features: pd.DataFrame
    instruments: pd.DataFrame | None


def assemble_causal_panel(
    entropy_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: CausalPanelConfig,
    *,
    join_on: Iterable[str] | None = None,
    how: str = "inner",
) -> pd.DataFrame:
    """Merge entropy metrics and market outcomes into a single panel.

    Args:
        entropy_df: Entropy metrics with columns referenced by ``config``.
        market_df: Market outcomes and controls.
        config: Panel column mapping.
        join_on: Optional override for join keys. Defaults to ``unit`` and
            ``time`` columns from ``config``.
        how: Merge strategy passed to :func:`pandas.DataFrame.merge`.

    Returns:
        A panel dataframe containing treatment, outcome, and covariates.

    Raises:
        KeyError: If required columns are missing in the source dataframes.
    """

    required_entropy = {config.unit_col, config.time_col, config.treatment_col}
    required_market = {config.unit_col, config.time_col, config.outcome_col}
    missing_entropy = required_entropy.difference(entropy_df.columns)
    missing_market = required_market.difference(market_df.columns)
    if missing_entropy:
        raise KeyError(
            f"Entropy dataframe missing required columns: {sorted(missing_entropy)}"
        )
    if missing_market:
        raise KeyError(
            f"Market dataframe missing required columns: {sorted(missing_market)}"
        )

    join_keys = list(join_on or (config.unit_col, config.time_col))
    panel = entropy_df.merge(market_df, on=join_keys, how=how, validate="many_to_one")
    selected_cols = set(join_keys) | {
        config.treatment_col,
        config.outcome_col,
        config.post_treatment_col,
    }
    if config.covariate_cols:
        selected_cols.update(config.covariate_cols)
    if config.instrument_cols:
        selected_cols.update(config.instrument_cols)
    missing_panel = selected_cols.difference(panel.columns)
    if missing_panel:
        raise KeyError(
            f"Merged panel missing required columns: {sorted(missing_panel)}"
        )
    ordered_cols = list(join_keys) + [
        config.treatment_col,
        config.post_treatment_col,
        config.outcome_col,
    ]
    if config.covariate_cols:
        ordered_cols.extend(config.covariate_cols)
    if config.instrument_cols:
        ordered_cols.extend(config.instrument_cols)
    return panel.loc[:, ordered_cols].sort_values(join_keys).reset_index(drop=True)


def build_propensity_features(
    panel: pd.DataFrame,
    config: CausalPanelConfig,
    *,
    window: int = 5,
) -> pd.DataFrame:
    """Create aggregated features for propensity score estimation.

    Args:
        panel: Full causal panel.
        config: Panel configuration.
        window: Rolling window length for summary statistics.

    Returns:
        Propensity feature dataframe indexed by unit and time columns.
    """

    time_col = config.time_col
    sorted_panel = panel.sort_values([config.unit_col, time_col])
    outcome_roll = (
        sorted_panel.groupby(config.unit_col)[config.outcome_col]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .rename("outcome_trend")
    )
    treatment_roll = (
        sorted_panel.groupby(config.unit_col)[config.treatment_col]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .rename("treatment_trend")
    )
    features = pd.DataFrame({
        config.unit_col: sorted_panel[config.unit_col],
        time_col: sorted_panel[time_col],
        "outcome_trend": outcome_roll.to_numpy(),
        "treatment_trend": treatment_roll.to_numpy(),
    })
    if config.covariate_cols:
        covariates = sorted_panel[list(config.covariate_cols)].add_prefix("cov_")
        features = pd.concat([features, covariates.reset_index(drop=True)], axis=1)
    return features


def prepare_causal_dataset(
    entropy_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: CausalPanelConfig,
    *,
    propensity_window: int = 5,
) -> CausalDataset:
    """Build the full :class:`CausalDataset` bundle."""

    panel = assemble_causal_panel(entropy_df, market_df, config)
    propensity = build_propensity_features(panel, config, window=propensity_window)
    instrument_cols = list(config.instrument_cols or [])
    instruments = panel[[config.unit_col, config.time_col] + instrument_cols] if instrument_cols else None
    return CausalDataset(panel=panel, propensity_features=propensity, instruments=instruments)
