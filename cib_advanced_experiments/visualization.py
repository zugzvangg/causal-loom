import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_synthetic_control_series_figure(
    df_to_plot: pd.DataFrame,
    treated_unit: str,
    treatment_date: pd.Timestamp,
    feature: str,
    control_group_via_matching: bool,
    loss_func: str,
    fit_error: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_to_plot["date"],
            y=df_to_plot["real"],
            mode="lines",
            name=treated_unit,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_to_plot["date"],
            y=df_to_plot["synthetic"],
            mode="lines",
            name="Synthetic Control",
        )
    )
    fig.add_vline(
        x=treatment_date.timestamp() * 1000,
        line=dict(dash="dash"),
        annotation_text="Treatment date",
    )
    title_addition = "using matching" if control_group_via_matching else "directly"
    fig.update_layout(
        title=(
            f"Synthetic Control for unit {treated_unit}.<br>"
            f"Control group is set {title_addition}.<br>"
            f"Fit error {loss_func} = {fit_error}"
        ),
        xaxis_title="Date",
        yaxis_title=feature,
        legend_title="Legend",
        font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
    )
    return fig


def create_synthetic_control_weights_figure(
    weights: np.ndarray,
    top_weights_to_plot: int = 5,
) -> go.Figure:
    top_weights_values = sorted(sorted(weights)[-top_weights_to_plot:], reverse=True)
    top_weights_values = [str(round(x, 4)) for x in top_weights_values]

    fig = px.histogram(
        x=weights,
        nbins=max(1, int(len(weights) / 3)),
        title=f"Top {top_weights_to_plot} weights: {'. '.join(top_weights_values)}",
    )
    fig.update_layout(
        xaxis_title="Weights",
        yaxis_title="Frequency",
        legend_title="Legend",
        font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
    )
    return fig


def create_experiment_distributions_figure(
    real: list[float],
    synthetic: list[float],
    fit_errors: list[float],
    border_quantile: float,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Synthetic vs Real values Distribution",
            "Fit Errors Distribution",
        ),
    )

    fig.add_trace(
        go.Histogram(
            x=real,
            nbinsx=max(1, int(len(real) / 2)),
            opacity=0.75,
            name="Real values <br>Distribution <br>and mean value",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=synthetic,
            nbinsx=max(1, int(len(synthetic) / 2)),
            opacity=0.75,
            name="Synthetic values <br>Distribution <br>and mean value",
        ),
        row=1,
        col=1,
    )

    fig.add_shape(
        type="line",
        x0=np.mean(synthetic),
        y0=0,
        x1=np.mean(synthetic),
        y1=10,
        xref="x",
        yref="paper",
        line=dict(color="Red", width=2, dash="dashdot"),
        row=1,
        col=1,
    )
    fig.add_shape(
        type="line",
        x0=np.mean(real),
        y0=0,
        x1=np.mean(real),
        y1=10,
        xref="x",
        yref="paper",
        line=dict(color="Blue", width=2, dash="dashdot"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=fit_errors,
            nbinsx=max(1, int(len(fit_errors) / 2)),
            opacity=1,
            name="Fit Errors <br>Distribution <br>and fit border quantile",
        ),
        row=1,
        col=2,
    )
    fig.add_shape(
        type="line",
        x0=border_quantile,
        y0=0,
        x1=border_quantile,
        y1=10,
        xref="x",
        yref="paper",
        line=dict(color="Green", width=2, dash="dashdot"),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text="Distributions",
        showlegend=True,
        bargap=0.1,
        width=1200,
        height=600,
        font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
    )
    return fig
