import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cib_advanced_experiments.visualization import (
    create_experiment_distributions_figure,
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


def test_create_experiment_distributions_figure_returns_figure():
    fig = create_experiment_distributions_figure(
        real=[1.0, 1.1, 1.2],
        synthetic=[0.9, 1.0, 1.1],
        fit_errors=[0.01, 0.02, 0.03],
        border_quantile=0.025,
    )
    assert isinstance(fig, go.Figure)
