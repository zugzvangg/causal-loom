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
    """Create line chart of treated real series vs synthetic control.

    Args:
        df_to_plot (pd.DataFrame): Dataframe with date, real, synthetic columns.
        treated_unit (str): Treated unit identifier.
        treatment_date (pd.Timestamp): Treatment date marker.
        feature (str): Outcome name for axis title.
        control_group_via_matching (bool): Whether controls were auto-matched.
        loss_func (str): Loss function name.
        fit_error (float): Model fit error value.

    Returns:
        go.Figure: Configured Plotly figure.
    """
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
    """Create histogram of donor weights.

    Args:
        weights (np.ndarray): Donor weights.
        top_weights_to_plot (int, optional): Number of top weights to display in title.

    Returns:
        go.Figure: Configured Plotly histogram.
    """
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


def create_synthetic_control_effect_figure(
    df_to_plot: pd.DataFrame,
    treated_unit: str,
    treatment_date: pd.Timestamp,
    feature: str,
) -> go.Figure:
    """Create effect/gap plot where effect equals real minus synthetic.

    Args:
        df_to_plot (pd.DataFrame): Dataframe with date, real, synthetic columns.
        treated_unit (str): Treated unit identifier.
        treatment_date (pd.Timestamp): Treatment date marker.
        feature (str): Outcome name for y-axis title.

    Returns:
        go.Figure: Configured Plotly figure.
    """
    effect_df = df_to_plot.copy()
    effect_df["effect"] = effect_df["real"] - effect_df["synthetic"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=effect_df["date"],
            y=effect_df["effect"],
            mode="lines",
            name="Estimated Effect",
        )
    )
    fig.add_hline(y=0.0, line=dict(color="black", width=2))
    fig.add_vline(
        x=pd.to_datetime(treatment_date).timestamp() * 1000,
        line=dict(color="black", dash="dot"),
        annotation_text="Treatment date",
    )
    fig.update_layout(
        title=f"Estimated Effect (Real - Synthetic) for {treated_unit}",
        xaxis_title="Date",
        yaxis_title=f"{feature} effect",
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
    """Create experiment-level distributions for outcomes and fit errors.

    Args:
        real (list[float]): Real treated values.
        synthetic (list[float]): Synthetic values.
        fit_errors (list[float]): Unit-level model fit errors.
        border_quantile (float): Fit error threshold value.

    Returns:
        go.Figure: Configured subplot figure.
    """
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


def create_experiment_aggregated_series_figure(
    df_to_plot: pd.DataFrame,
    feature: str,
    treated_units_count: int,
    treatment_date: pd.Timestamp | None = None,
) -> go.Figure:
    """Create mean real-vs-synthetic series across fitted treated units.

    Args:
        df_to_plot (pd.DataFrame): Dataframe with date, real, synthetic columns.
        feature (str): Outcome name for axis title.
        treated_units_count (int): Number of treated units included in the mean.
        treatment_date (pd.Timestamp | None, optional): Common treatment date marker.

    Returns:
        go.Figure: Configured Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_to_plot["date"],
            y=df_to_plot["real"],
            mode="lines",
            name="Mean real",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_to_plot["date"],
            y=df_to_plot["synthetic"],
            mode="lines",
            name="Mean synthetic",
        )
    )
    if treatment_date is not None:
        fig.add_vline(
            x=pd.to_datetime(treatment_date).timestamp() * 1000,
            line=dict(dash="dash"),
            annotation_text="Treatment date",
        )
    fig.update_layout(
        title=(
            "Mean Synthetic Control Across Treated Units.<br>"
            f"Units included: {treated_units_count}"
        ),
        xaxis_title="Date",
        yaxis_title=feature,
        legend_title="Legend",
        font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
    )
    return fig


def create_placebo_gap_figure(
    placebo_effects: pd.DataFrame,
    treatment_date: pd.Timestamp,
    actual_treated_unit: str,
) -> go.Figure:
    """Create placebo gap-over-time chart with highlighted treated unit.

    Args:
        placebo_effects (pd.DataFrame): Dataframe with treated_unit, date and effect columns.
        treatment_date (pd.Timestamp): Treatment date marker.
        actual_treated_unit (str): Actual treated unit identifier for highlighting.

    Returns:
        go.Figure: Configured Plotly figure.
    """
    fig = go.Figure()
    for unit in sorted(placebo_effects["treated_unit"].astype(str).unique().tolist()):
        unit_data = placebo_effects[
            placebo_effects["treated_unit"].astype(str) == str(unit)
        ]
        is_actual = str(unit) == str(actual_treated_unit)
        fig.add_trace(
            go.Scatter(
                x=unit_data["date"],
                y=unit_data["effect"],
                mode="lines",
                name=str(unit) if is_actual else "Placebo",
                line=dict(color="royalblue" if is_actual else "rgba(120,120,120,0.35)"),
                showlegend=is_actual,
            )
        )
    fig.add_hline(y=0.0, line=dict(color="black", width=2))
    fig.add_vline(
        x=pd.to_datetime(treatment_date).timestamp() * 1000,
        line=dict(color="black", dash="dot"),
        annotation_text="Treatment date",
    )
    fig.update_layout(
        title="Placebo Effects Over Time (Gap = Real - Synthetic)",
        xaxis_title="Date",
        yaxis_title="Effect",
        legend_title="Legend",
        font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
    )
    return fig


def create_placebo_distribution_figure(
    effect_values: list[float],
    treated_effect: float,
    effect_evaluation_aggregation: str,
) -> go.Figure:
    """Create placebo effect distribution with highlighted treated effect.

    Args:
        effect_values (list[float]): Placebo/reference unit-level effect values.
        treated_effect (float): Treated-unit effect value.
        effect_evaluation_aggregation (str): Aggregation label used for effect values.

    Returns:
        go.Figure: Configured Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=effect_values,
            nbinsx=max(1, int(len(effect_values) / 2)),
            opacity=0.7,
            name="Placebo effects",
        )
    )
    fig.add_vline(
        x=treated_effect,
        line=dict(color="royalblue", width=3, dash="dash"),
        annotation_text="Treated effect",
    )
    fig.update_layout(
        title=f"Placebo Effect Distribution ({effect_evaluation_aggregation})",
        xaxis_title="Effect",
        yaxis_title="Frequency",
        legend_title="Legend",
        font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
    )
    return fig


def create_conformal_residuals_figure(
    residuals_df: pd.DataFrame,
    intervention_start: pd.Timestamp,
) -> go.Figure:
    """Create conformal residuals time-series plot.

    Args:
        residuals_df (pd.DataFrame): Dataframe with date and residuals.
        intervention_start (pd.Timestamp): Intervention date marker.

    Returns:
        go.Figure: Configured Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=residuals_df["date"],
            y=residuals_df["residuals"],
            mode="lines",
            name="Estimated Effect (Residuals)",
        )
    )
    fig.add_hline(y=0.0, line=dict(color="black", width=2))
    fig.add_vline(
        x=pd.to_datetime(intervention_start).timestamp() * 1000,
        line=dict(color="black", dash="dot"),
        annotation_text="Intervention",
    )
    fig.update_layout(
        title="Conformal Residuals Under Null",
        xaxis_title="Date",
        yaxis_title="Residual",
        font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
    )
    return fig


def create_conformal_p_value_grid_figure(
    p_values_df: pd.DataFrame,
    alpha: float = 0.1,
) -> go.Figure:
    """Create p-value versus null-effect grid for one period.

    Args:
        p_values_df (pd.DataFrame): One-column p-value grid indexed by null effects.
        alpha (float, optional): Significance level marker.

    Returns:
        go.Figure: Configured Plotly figure.
    """
    period = p_values_df.columns[0]
    ci_candidates = p_values_df[p_values_df[period] >= alpha]
    ci_low = ci_candidates.index.min() if not ci_candidates.empty else None
    ci_high = ci_candidates.index.max() if not ci_candidates.empty else None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=p_values_df[period].values,
            y=p_values_df.index.values,
            mode="lines",
            name="P-value curve",
        )
    )
    fig.add_vline(
        x=alpha,
        line=dict(color="black", dash="dot"),
        annotation_text=f"alpha={alpha}",
    )
    if ci_low is not None and ci_high is not None:
        fig.add_hline(y=ci_low, line=dict(color="orange", dash="dash"))
        fig.add_hline(y=ci_high, line=dict(color="orange", dash="dash"))

    fig.update_layout(
        title=f"Conformal P-Value Grid ({pd.to_datetime(period).date()})",
        xaxis_title="P-value",
        yaxis_title="H0 (effect)",
        font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
    )
    return fig


def create_conformal_ci_band_figure(
    effect_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    treatment_date: pd.Timestamp,
    alpha: float,
) -> go.Figure:
    """Create effect plot with conformal confidence interval band.

    Args:
        effect_df (pd.DataFrame): Dataframe with date and effect columns.
        ci_df (pd.DataFrame): Confidence interval dataframe by date.
        treatment_date (pd.Timestamp): Intervention date marker.
        alpha (float): Significance level used to determine CI column names.

    Returns:
        go.Figure: Configured Plotly figure.
    """
    ci_level = int(100 * (1 - alpha))
    lower_col = f"{ci_level}_ci_lower"
    upper_col = f"{ci_level}_ci_upper"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=effect_df["date"],
            y=effect_df["effect"],
            mode="lines",
            name="Estimated Effect",
            line=dict(color="orange"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ci_df.index,
            y=ci_df[upper_col],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ci_df.index,
            y=ci_df[lower_col],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255,165,0,0.2)",
            line=dict(width=0),
            name=f"{ci_level}% CI",
        )
    )
    fig.add_hline(y=0.0, line=dict(color="black", width=2))
    fig.add_vline(
        x=pd.to_datetime(treatment_date).timestamp() * 1000,
        line=dict(color="black", dash="dot"),
        annotation_text="Intervention",
    )
    fig.update_layout(
        title="Conformal Confidence Interval Over Time",
        xaxis_title="Date",
        yaxis_title="Effect",
        font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
    )
    return fig


def create_conformal_dashboard_figure(
    residuals_df: pd.DataFrame,
    p_values_df: pd.DataFrame,
    effect_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    treatment_date: pd.Timestamp,
    alpha: float = 0.1,
) -> go.Figure:
    """Create unified conformal dashboard with three panels.

    Args:
        residuals_df (pd.DataFrame): Residuals dataframe.
        p_values_df (pd.DataFrame): P-value grid dataframe for one period.
        effect_df (pd.DataFrame): Effect dataframe with date/effect columns.
        ci_df (pd.DataFrame): Confidence interval dataframe by date.
        treatment_date (pd.Timestamp): Intervention date marker.
        alpha (float, optional): Significance level used in p-grid and CI labels.

    Returns:
        go.Figure: Configured dashboard figure.
    """
    ci_level = int(100 * (1 - alpha))
    lower_col = f"{ci_level}_ci_lower"
    upper_col = f"{ci_level}_ci_upper"
    period = p_values_df.columns[0]
    ci_candidates = p_values_df[p_values_df[period] >= alpha]
    ci_low = ci_candidates.index.min() if not ci_candidates.empty else None
    ci_high = ci_candidates.index.max() if not ci_candidates.empty else None

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy", "colspan": 2}, None]],
        subplot_titles=(
            "Conformal Residuals",
            "P-value Grid",
            "Effect With Conformal CI",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=residuals_df["date"],
            y=residuals_df["residuals"],
            mode="lines",
            name="Residuals",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0.0, line=dict(color="black", width=2), row=1, col=1)
    fig.add_vline(
        x=pd.to_datetime(treatment_date).timestamp() * 1000,
        line=dict(color="black", dash="dot"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=p_values_df[period].values,
            y=p_values_df.index.values,
            mode="lines",
            name="P-value curve",
        ),
        row=1,
        col=2,
    )
    fig.add_vline(x=alpha, line=dict(color="black", dash="dot"), row=1, col=2)
    if ci_low is not None and ci_high is not None:
        fig.add_hline(y=ci_low, line=dict(color="orange", dash="dash"), row=1, col=2)
        fig.add_hline(y=ci_high, line=dict(color="orange", dash="dash"), row=1, col=2)

    fig.add_trace(
        go.Scatter(
            x=effect_df["date"],
            y=effect_df["effect"],
            mode="lines",
            name="Estimated Effect",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ci_df.index,
            y=ci_df[upper_col],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ci_df.index,
            y=ci_df[lower_col],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255,165,0,0.2)",
            line=dict(width=0),
            name=f"{ci_level}% CI",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line=dict(color="black", width=2), row=2, col=1)
    fig.add_vline(
        x=pd.to_datetime(treatment_date).timestamp() * 1000,
        line=dict(color="black", dash="dot"),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Conformal Inference Dashboard",
        height=900,
        width=1300,
        font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
    )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=1, col=1)
    fig.update_xaxes(title_text="P-value", row=1, col=2)
    fig.update_yaxes(title_text="H0 (effect)", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Effect", row=2, col=1)
    return fig
