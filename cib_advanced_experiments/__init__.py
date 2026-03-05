"""Top-level package exports."""

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
    create_experiment_distributions_figure,
    create_synthetic_control_series_figure,
    create_synthetic_control_weights_figure,
)


__all__ = [
    "build_conformal_frame",
    "ConformalFrame",
    "Experiment",
    "MatchingResult",
    "PropensityScoreMatchingResult",
    "prepare_panel_data",
    "prepare_rdd_data",
    "propensity_score_match",
    "select_control_group_by_propensity_from_panel",
    "select_control_group_by_features",
    "create_experiment_distributions_figure",
    "create_synthetic_control_series_figure",
    "create_synthetic_control_weights_figure",
    "SynthControlModel",
    "SyntheticControl",
]
