"""Causal inference tooling for Entropy News research workflows."""

from .data import (
    CausalDataset,
    CausalPanelConfig,
    assemble_causal_panel,
    build_propensity_features,
    prepare_causal_dataset,
)
from .models import (
    DifferenceInDifferencesResult,
    SyntheticControlResult,
    TwoStageLeastSquaresResult,
    difference_in_differences,
    synthetic_control,
    two_stage_least_squares,
)
from .reporting import (
    PolicyScenario,
    build_summary_table,
    format_policy_narrative,
    prepare_counterfactual_series,
)

__all__ = [
    "CausalDataset",
    "CausalPanelConfig",
    "assemble_causal_panel",
    "build_propensity_features",
    "prepare_causal_dataset",
    "DifferenceInDifferencesResult",
    "SyntheticControlResult",
    "TwoStageLeastSquaresResult",
    "difference_in_differences",
    "synthetic_control",
    "two_stage_least_squares",
    "PolicyScenario",
    "build_summary_table",
    "format_policy_narrative",
    "prepare_counterfactual_series",
]
