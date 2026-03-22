import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cib_advanced_experiments.visualization import (
    create_conformal_ci_band_figure,
    create_conformal_dashboard_figure,
    create_conformal_p_value_grid_figure,
    create_conformal_residuals_decomposition_figure,
    create_conformal_residuals_figure,
    create_experiment_aggregated_series_figure,
    create_did_2x2_figure,
    create_did_parallel_trends_figure,
    create_experiment_distributions_figure,
    create_placebo_distribution_figure,
    create_placebo_gap_figure,
    create_synthetic_control_effect_figure,
    create_synthetic_control_series_figure,
    create_synthetic_control_weights_figure,
)


def test_create_synthetic_control_series_figure_returns_figure():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "real": [1.0, 1.1],
            "synthetic": [0.9, 1.0],
        }
    )
    fig = create_synthetic_control_series_figure(
        df_to_plot=df,
        treated_unit="u1",
        treatment_date=pd.Timestamp("2024-01-02"),
        feature="y",
        control_group_via_matching=True,
        loss_func="mse",
        fit_error=0.12,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_create_synthetic_control_weights_figure_returns_figure():
    fig = create_synthetic_control_weights_figure(
        weights=np.array([0.1, 0.3, 0.6]),
        top_weights_to_plot=2,
    )
    assert isinstance(fig, go.Figure)


def test_create_synthetic_control_effect_figure_returns_figure():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "real": [1.0, 1.2],
            "synthetic": [0.9, 1.0],
        }
    )
    fig = create_synthetic_control_effect_figure(
        df_to_plot=df,
        treated_unit="u1",
        treatment_date=pd.Timestamp("2024-01-02"),
        feature="y",
    )
    assert isinstance(fig, go.Figure)


def test_create_experiment_distributions_figure_returns_figure():
    fig = create_experiment_distributions_figure(
        real=[1.0, 1.1, 1.2],
        synthetic=[0.9, 1.0, 1.1],
        fit_errors=[0.01, 0.02, 0.03],
        border_quantile=0.025,
    )
    assert isinstance(fig, go.Figure)


def test_create_experiment_aggregated_series_figure_returns_figure():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "real": [1.0, 1.1],
            "synthetic": [0.9, 1.0],
            "effect": [0.1, 0.1],
        }
    )
    fig = create_experiment_aggregated_series_figure(
        df_to_plot=df,
        feature="y",
        treated_units_count=3,
        treatment_date=pd.Timestamp("2024-01-02"),
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_create_placebo_gap_figure_returns_figure():
    df = pd.DataFrame(
        {
            "treated_unit": ["u0", "u0", "u1", "u1"],
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]
            ),
            "effect": [0.1, 0.2, -0.05, 0.03],
        }
    )
    fig = create_placebo_gap_figure(
        placebo_effects=df,
        treatment_date=pd.Timestamp("2024-01-02"),
        actual_treated_unit="u0",
    )
    assert isinstance(fig, go.Figure)


def test_create_placebo_distribution_figure_returns_figure():
    fig = create_placebo_distribution_figure(
        effect_values=[-0.1, 0.0, 0.05, 0.2],
        treated_effect=0.12,
        effect_evaluation_aggregation="mean_value",
    )
    assert isinstance(fig, go.Figure)


def test_create_did_2x2_figure_returns_figure():
    group_means = pd.DataFrame(
        {
            "group": ["control", "control", "treated", "treated"],
            "post_period": [0, 1, 0, 1],
            "mean_outcome": [1.0, 1.2, 1.1, 1.7],
        }
    )
    fig = create_did_2x2_figure(
        group_means=group_means,
        counterfactual_treated_post=1.3,
        estimate=0.4,
    )
    assert isinstance(fig, go.Figure)


def test_create_did_parallel_trends_figure_returns_figure():
    trends = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "treated_mean": [1.0, 1.05, 1.1],
            "control_mean": [0.9, 0.95, 1.0],
        }
    )
    fig = create_did_parallel_trends_figure(
        trends_frame=trends,
        treatment_start=pd.Timestamp("2024-01-03"),
        outcome_column="y",
        pre_period_only=False,
    )
    assert isinstance(fig, go.Figure)


def test_create_conformal_residuals_figure_returns_figure():
    residuals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "residuals": [0.1, -0.05, 0.02],
        }
    )
    fig = create_conformal_residuals_figure(
        residuals_df=residuals,
        intervention_start=pd.Timestamp("2024-01-02"),
    )
    assert isinstance(fig, go.Figure)


def test_create_conformal_residuals_decomposition_figure_returns_figure():
    residuals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y0": [1.0, 1.1, 1.2],
            "y0_est": [0.9, 1.0, 1.05],
            "residuals": [0.1, 0.1, 0.15],
        }
    )
    fig = create_conformal_residuals_decomposition_figure(
        residuals_df=residuals,
        intervention_start=pd.Timestamp("2024-01-02"),
    )
    assert isinstance(fig, go.Figure)


def test_create_conformal_p_value_grid_figure_returns_figure():
    p_values = pd.DataFrame(
        {
            pd.Timestamp("2024-01-03"): {
                -0.2: 0.03,
                -0.1: 0.20,
                0.0: 0.42,
                0.1: 0.12,
                0.2: 0.04,
            }
        }
    )
    fig = create_conformal_p_value_grid_figure(p_values_df=p_values, alpha=0.1)
    assert isinstance(fig, go.Figure)


def test_create_conformal_ci_band_figure_returns_figure():
    effect_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "effect": [0.05, 0.08],
        }
    )
    ci_df = pd.DataFrame(
        {
            "90_ci_lower": [-0.1, -0.05],
            "90_ci_upper": [0.2, 0.22],
        },
        index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
    )
    fig = create_conformal_ci_band_figure(
        effect_df=effect_df,
        ci_df=ci_df,
        treatment_date=pd.Timestamp("2024-01-03"),
        alpha=0.1,
    )
    assert isinstance(fig, go.Figure)


def test_create_conformal_dashboard_figure_returns_figure():
    residuals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "residuals": [0.1, -0.05, 0.02],
        }
    )
    p_values = pd.DataFrame(
        {
            pd.Timestamp("2024-01-03"): {
                -0.2: 0.03,
                -0.1: 0.20,
                0.0: 0.42,
                0.1: 0.12,
                0.2: 0.04,
            }
        }
    )
    effect_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "effect": [0.05, 0.08],
        }
    )
    ci_df = pd.DataFrame(
        {
            "90_ci_lower": [-0.1, -0.05],
            "90_ci_upper": [0.2, 0.22],
        },
        index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
    )
    fig = create_conformal_dashboard_figure(
        residuals_df=residuals,
        p_values_df=p_values,
        effect_df=effect_df,
        ci_df=ci_df,
        treatment_date=pd.Timestamp("2024-01-03"),
        alpha=0.1,
    )
    assert isinstance(fig, go.Figure)
