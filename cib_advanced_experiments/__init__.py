"""Top-level package exports."""

from cib_advanced_experiments.conformal_inference import (
    conformal_confidence_interval,
    conformal_confidence_interval_from_p_values,
    conformal_p_value,
    conformal_p_value_from_residuals,
    conformal_p_value_grid_for_period,
    conformal_residuals,
    conformal_test_statistic,
)
from cib_advanced_experiments.experiment import Experiment
from cib_advanced_experiments.matching import MatchingResult, select_control_group_by_features
from cib_advanced_experiments.model import SynthControlModel
from cib_advanced_experiments.preprocessing import (
    ConformalFrame,
    build_conformal_frame,
    prepare_panel_data,
    prepare_rdd_data,
)
from cib_advanced_experiments.propensity_score_matching import (
    PropensityScoreMatchingResult,
    propensity_score_match,
    select_control_group_by_propensity_from_panel,
)
from cib_advanced_experiments.synthetic_control import SyntheticControl
from cib_advanced_experiments.visualization import (
    create_conformal_ci_band_figure,
    create_conformal_dashboard_figure,
    create_conformal_p_value_grid_figure,
    create_conformal_residuals_figure,
    create_experiment_distributions_figure,
    create_synthetic_control_effect_figure,
    create_synthetic_control_series_figure,
    create_synthetic_control_weights_figure,
)


__all__ = [
    "build_conformal_frame",
    "conformal_confidence_interval",
    "conformal_confidence_interval_from_p_values",
    "conformal_p_value",
    "conformal_p_value_from_residuals",
    "conformal_p_value_grid_for_period",
    "conformal_residuals",
    "conformal_test_statistic",
    "ConformalFrame",
    "Experiment",
    "MatchingResult",
    "PropensityScoreMatchingResult",
    "prepare_panel_data",
    "prepare_rdd_data",
    "propensity_score_match",
    "select_control_group_by_propensity_from_panel",
    "select_control_group_by_features",
    "create_conformal_ci_band_figure",
    "create_conformal_dashboard_figure",
    "create_conformal_p_value_grid_figure",
    "create_conformal_residuals_figure",
    "create_experiment_distributions_figure",
    "create_synthetic_control_effect_figure",
    "create_synthetic_control_series_figure",
    "create_synthetic_control_weights_figure",
    "SynthControlModel",
    "SyntheticControl",
]
