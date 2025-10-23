"""Tests for the causal analysis research toolkit."""

from __future__ import annotations

import pandas as pd
import pytest

from entropy_news.research.causal import (
    CausalDataset,
    CausalPanelConfig,
    PolicyScenario,
    assemble_causal_panel,
    build_propensity_features,
    build_summary_table,
    difference_in_differences,
    format_policy_narrative,
    prepare_causal_dataset,
    prepare_counterfactual_series,
    synthetic_control,
    two_stage_least_squares,
)


@pytest.fixture()
def sample_frames() -> tuple[pd.DataFrame, pd.DataFrame, CausalPanelConfig]:
    """Construct toy entropy and market frames for testing."""

    records: list[dict[str, float | str | int]] = []
    market_records: list[dict[str, float | str | int]] = []
    units = ["treated", "control_a", "control_b"]
    for time in range(3):
        post = 1 if time >= 1 else 0
        for unit in units:
            treat = 1 if unit == "treated" else 0
            base = 10 + 0.8 * time + (0.2 if unit == "control_b" else 0)
            effect = 3.0 if treat else 0.0
            unit_noise = {"treated": 0.2, "control_a": -0.1, "control_b": 0.05}[unit]
            outcome = base + effect + unit_noise * time
            entropy_signal = 0.3 * time + (0.5 if unit == "treated" else 0.1 * time)
            instrument = 0.4 + 0.8 * post if unit == "treated" else 0.2 + 0.1 * time
            volatility = 0.5 + 0.15 * time + (0.1 if unit == "treated" else 0.05 * post)
            records.append(
                {
                    "unit": unit,
                    "time": time,
                    "treatment": treat,
                    "post": post,
                    "entropy": entropy_signal,
                    "instrument": instrument,
                }
            )
            market_records.append(
                {
                    "unit": unit,
                    "time": time,
                    "outcome": outcome,
                    "volatility": volatility,
                }
            )
    entropy_df = pd.DataFrame.from_records(records)
    market_df = pd.DataFrame.from_records(market_records)
    config = CausalPanelConfig(
        unit_col="unit",
        time_col="time",
        outcome_col="outcome",
        treatment_col="treatment",
        post_treatment_col="post",
        covariate_cols=("volatility",),
        instrument_cols=("instrument",),
    )
    return entropy_df, market_df, config


def test_assemble_panel_and_propensity(sample_frames: tuple[pd.DataFrame, pd.DataFrame, CausalPanelConfig]) -> None:
    entropy_df, market_df, config = sample_frames
    panel = assemble_causal_panel(entropy_df, market_df, config)
    assert list(panel.columns[:6]) == [
        config.unit_col,
        config.time_col,
        config.treatment_col,
        config.post_treatment_col,
        config.outcome_col,
        "volatility",
    ]
    features = build_propensity_features(panel, config, window=2)
    assert {"outcome_trend", "treatment_trend"}.issubset(features.columns)
    assert len(features) == len(panel)


def test_prepare_causal_dataset_bundle(sample_frames: tuple[pd.DataFrame, pd.DataFrame, CausalPanelConfig]) -> None:
    entropy_df, market_df, config = sample_frames
    bundle = prepare_causal_dataset(entropy_df, market_df, config, propensity_window=2)
    assert isinstance(bundle, CausalDataset)
    assert bundle.panel.equals(assemble_causal_panel(entropy_df, market_df, config))
    assert set(bundle.propensity_features.columns).issuperset({"outcome_trend", "treatment_trend"})
    assert bundle.instruments is not None
    assert set(bundle.instruments.columns) == {config.unit_col, config.time_col, "instrument"}


def test_difference_in_differences(sample_frames: tuple[pd.DataFrame, pd.DataFrame, CausalPanelConfig]) -> None:
    entropy_df, market_df, config = sample_frames
    panel = assemble_causal_panel(entropy_df, market_df, config)
    did_result = difference_in_differences(panel, config)
    treated = panel[panel[config.treatment_col] == 1]
    control = panel[panel[config.treatment_col] == 0]
    expected_att = (
        treated[treated[config.post_treatment_col] == 1][config.outcome_col].mean()
        - treated[treated[config.post_treatment_col] == 0][config.outcome_col].mean()
        - (
            control[control[config.post_treatment_col] == 1][config.outcome_col].mean()
            - control[control[config.post_treatment_col] == 0][config.outcome_col].mean()
        )
    )
    assert did_result.average_treatment_effect == pytest.approx(expected_att, abs=1e-6)
    assert did_result.standard_error >= 0
    assert did_result.confidence_interval[0] < did_result.confidence_interval[1]


def test_two_stage_least_squares(sample_frames: tuple[pd.DataFrame, pd.DataFrame, CausalPanelConfig]) -> None:
    entropy_df, market_df, config = sample_frames
    panel = assemble_causal_panel(entropy_df, market_df, config)
    tsls = two_stage_least_squares(panel, config)
    assert tsls.coefficient == pytest.approx(3.0, rel=0.2)
    assert tsls.first_stage_f_stat > 1.0
    assert tsls.residual_variance >= 0


def test_synthetic_control_and_reporting(sample_frames: tuple[pd.DataFrame, pd.DataFrame, CausalPanelConfig]) -> None:
    entropy_df, market_df, config = sample_frames
    panel = assemble_causal_panel(entropy_df, market_df, config)
    sc_result = synthetic_control(panel, config, treated_unit="treated", donor_units=["control_a", "control_b"])
    summary = build_summary_table(
        difference_in_differences(panel, config),
        iv_result=two_stage_least_squares(panel, config),
        sc_result=sc_result,
    )
    assert set(summary["estimator"]) == {
        "difference_in_differences",
        "two_stage_least_squares",
        "synthetic_control",
    }
    scenario = PolicyScenario(
        name="Policy Tightening",
        description="Reduced liquidity injections",
        target_group="systemically important institutions",
    )
    narrative = format_policy_narrative(scenario, difference_in_differences(panel, config))
    assert "Policy Tightening" in narrative
    series = prepare_counterfactual_series(sc_result)
    assert set(series.columns) == {"time", "actual", "synthetic", "effect"}
    trimmed = prepare_counterfactual_series(sc_result, horizon=[0, 2])
    assert trimmed["time"].tolist() == [0, 2]
