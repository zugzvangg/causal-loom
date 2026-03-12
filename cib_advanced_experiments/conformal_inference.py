import datetime
from typing import Iterable, Union

import numpy as np
import pandas as pd

from cib_advanced_experiments.model import SynthControlModel

DateLike = Union[str, datetime.datetime, pd.Timestamp]
NullLike = Union[float, int, np.ndarray, list[float]]


def _prepare_wide_panel(
    panel_data: pd.DataFrame,
    unit_column: str,
    date_column: str,
    outcome_column: str,
    treated_unit: Union[str, int],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Pivot panel data to wide format and validate treated unit availability.

    Args:
        panel_data (pd.DataFrame): Long-format panel dataset.
        unit_column (str): Unit identifier column.
        date_column (str): Time column.
        outcome_column (str): Outcome variable column.
        treated_unit (Union[str, int]): Treated unit identifier.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: Wide dataframe and sorted date index as ndarray.
    """
    treated_unit = str(treated_unit)
    data = panel_data[[unit_column, date_column, outcome_column]].copy()
    data[unit_column] = data[unit_column].astype(str)
    unit_order = data[unit_column].drop_duplicates().tolist()
    data[date_column] = pd.to_datetime(data[date_column])

    wide = data.pivot(index=date_column, columns=unit_column, values=outcome_column)
    wide = wide.sort_index()
    ordered_columns = [unit for unit in unit_order if unit in wide.columns]
    wide = wide.reindex(columns=ordered_columns)
    if treated_unit not in wide.columns:
        raise ValueError(f"Treated unit '{treated_unit}' is not present in panel_data.")
    return wide, wide.index.to_numpy()


def _resolve_window_mask(
    dates: np.ndarray,
    intervention_start: DateLike,
    window: int = None,
) -> np.ndarray:
    """Build a post-intervention mask, optionally capped by window size.

    Args:
        dates (np.ndarray): Sorted date values.
        intervention_start (DateLike): Intervention start date.
        window (int, optional): Number of post-treatment periods to include.

    Returns:
        np.ndarray: Boolean mask for selected post-treatment rows.
    """
    intervention_start = pd.to_datetime(intervention_start)
    if window is None:
        mask = dates >= intervention_start
    else:
        if window <= 0:
            raise ValueError(f"window should be positive; got {window}.")
        post_dates = dates[dates >= intervention_start]
        if len(post_dates) == 0:
            raise ValueError(
                "No dates available at or after intervention_start for conformal inference."
            )
        last_date = post_dates[min(window, len(post_dates)) - 1]
        mask = (dates >= intervention_start) & (dates <= last_date)
    return mask


def _null_vector(null_hypothesis: NullLike, size: int) -> np.ndarray:
    """Normalize scalar/vector null hypothesis into a 1D vector.

    Args:
        null_hypothesis (NullLike): Scalar null effect or effect trajectory.
        size (int): Required output length.

    Returns:
        np.ndarray: Null effect vector with shape (size,).
    """
    if np.isscalar(null_hypothesis):
        return np.repeat(float(null_hypothesis), size)
    values = np.asarray(null_hypothesis, dtype=float).reshape(-1)
    if values.shape[0] != size:
        raise ValueError(
            f"Null hypothesis trajectory length should be {size}; got {values.shape[0]}."
        )
    return values


def conformal_residuals(
    panel_data: pd.DataFrame,
    unit_column: str,
    date_column: str,
    outcome_column: str,
    treated_unit: Union[str, int],
    intervention_start: DateLike,
    null_hypothesis: NullLike,
    model_kwargs: dict,
    window: int = None,
) -> pd.DataFrame:
    """Compute residuals under a null hypothesis by refitting synthetic control.

    Args:
        panel_data (pd.DataFrame): Long-format panel dataset.
        unit_column (str): Unit identifier column.
        date_column (str): Time column.
        outcome_column (str): Outcome variable.
        treated_unit (Union[str, int]): Treated unit identifier.
        intervention_start (DateLike): Intervention start date.
        null_hypothesis (NullLike): Null effect (scalar or vector for post window).
        model_kwargs (dict): Arguments for SynthControlModel.
        window (int, optional): Number of post-treatment periods to include.

    Returns:
        pd.DataFrame: Residual dataframe with date, y0, y0_est, residuals and mask columns.
    """
    wide, dates = _prepare_wide_panel(
        panel_data=panel_data,
        unit_column=unit_column,
        date_column=date_column,
        outcome_column=outcome_column,
        treated_unit=treated_unit,
    )
    treated_unit = str(treated_unit)
    post_mask = _resolve_window_mask(
        dates=dates, intervention_start=intervention_start, window=window
    )
    post_count = int(post_mask.sum())
    if post_count == 0:
        raise ValueError("Post-intervention window is empty.")

    y_observed = wide[treated_unit].to_numpy(dtype=float)
    y_under_null = y_observed.copy()
    y_under_null[post_mask] = y_under_null[post_mask] - _null_vector(
        null_hypothesis, post_count
    )

    X = wide.drop(columns=treated_unit).to_numpy(dtype=float)
    model = SynthControlModel(**model_kwargs)
    model.fit(X, y_under_null)
    weights = model.get_weights()
    y0_est = X.dot(weights)
    residuals = y_under_null - y0_est

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "y0": y_under_null,
            "y0_est": y0_est,
            "residuals": residuals,
            "post_intervention": post_mask,
        }
    )
    if window is not None:
        cutoff = result.loc[result["post_intervention"], "date"].max()
        result = result[result["date"] <= cutoff].copy()
    return result.reset_index(drop=True)


def conformal_test_statistic(u_hat: np.ndarray, q: int = 1, axis: int = 0) -> float:
    """Compute L_q conformal test statistic.

    Args:
        u_hat (np.ndarray): Residual block matrix or vector.
        q (int, optional): Norm power parameter.
        axis (int, optional): Aggregation axis.

    Returns:
        float: Test statistic value.
    """
    if q <= 0:
        raise ValueError(f"q should be positive; got {q}.")
    return (np.abs(u_hat) ** q).mean(axis=axis) ** (1 / q)


def conformal_p_value_from_residuals(residuals_df: pd.DataFrame, q: int = 1) -> float:
    """Compute permutation conformal p-value from residual dataframe.

    Args:
        residuals_df (pd.DataFrame): Output of conformal_residuals.
        q (int, optional): Norm power parameter for test statistic.

    Returns:
        float: Conformal p-value.
    """
    u = residuals_df["residuals"].to_numpy()
    post_mask = residuals_df["post_intervention"].to_numpy()
    block_permutations = np.stack(
        [np.roll(u, shift=i, axis=0)[post_mask] for i in range(len(u))]
    )
    statistics = conformal_test_statistic(block_permutations, q=q, axis=1)
    return float(np.mean(statistics >= statistics[0]))


def conformal_p_value(
    panel_data: pd.DataFrame,
    unit_column: str,
    date_column: str,
    outcome_column: str,
    treated_unit: Union[str, int],
    intervention_start: DateLike,
    null_hypothesis: NullLike,
    model_kwargs: dict,
    window: int = None,
    q: int = 1,
) -> float:
    """Compute conformal p-value for a null hypothesis from panel data.

    Args:
        panel_data (pd.DataFrame): Long-format panel dataset.
        unit_column (str): Unit identifier column.
        date_column (str): Time column.
        outcome_column (str): Outcome variable.
        treated_unit (Union[str, int]): Treated unit identifier.
        intervention_start (DateLike): Intervention start date.
        null_hypothesis (NullLike): Null effect.
        model_kwargs (dict): Arguments for SynthControlModel.
        window (int, optional): Number of post-treatment periods to include.
        q (int, optional): Norm power parameter.

    Returns:
        float: Conformal p-value.
    """
    resid = conformal_residuals(
        panel_data=panel_data,
        unit_column=unit_column,
        date_column=date_column,
        outcome_column=outcome_column,
        treated_unit=treated_unit,
        intervention_start=intervention_start,
        null_hypothesis=null_hypothesis,
        model_kwargs=model_kwargs,
        window=window,
    )
    return conformal_p_value_from_residuals(resid, q=q)


def conformal_p_value_grid_for_period(
    panel_data: pd.DataFrame,
    unit_column: str,
    date_column: str,
    outcome_column: str,
    treated_unit: Union[str, int],
    intervention_start: DateLike,
    period: DateLike,
    nulls: Iterable[float],
    model_kwargs: dict,
    q: int = 1,
) -> pd.DataFrame:
    """Evaluate p-value grid over candidate null effects for one period.

    Args:
        panel_data (pd.DataFrame): Long-format panel dataset.
        unit_column (str): Unit identifier column.
        date_column (str): Time column.
        outcome_column (str): Outcome variable.
        treated_unit (Union[str, int]): Treated unit identifier.
        intervention_start (DateLike): Global intervention start date.
        period (DateLike): Target post-treatment period for pointwise inference.
        nulls (Iterable[float]): Candidate null effects.
        model_kwargs (dict): Arguments for SynthControlModel.
        q (int, optional): Norm power parameter.

    Returns:
        pd.DataFrame: One-column dataframe indexed by null values with p-values.
    """
    intervention_start = pd.to_datetime(intervention_start)
    period = pd.to_datetime(period)
    data = panel_data.copy()
    data[date_column] = pd.to_datetime(data[date_column])

    aug_data = data[(data[date_column] < intervention_start) | (data[date_column] == period)]
    p_values = {}
    for null_value in nulls:
        p_values[float(null_value)] = conformal_p_value(
            panel_data=aug_data,
            unit_column=unit_column,
            date_column=date_column,
            outcome_column=outcome_column,
            treated_unit=treated_unit,
            intervention_start=period,
            null_hypothesis=float(null_value),
            model_kwargs=model_kwargs,
            window=1,
            q=q,
        )
    return pd.DataFrame({period: p_values}).sort_index()


def conformal_confidence_interval_from_p_values(
    p_values: pd.DataFrame,
    alpha: float = 0.1,
) -> pd.DataFrame:
    """Invert p-value grid into a (1-alpha) confidence interval.

    Args:
        p_values (pd.DataFrame): One-period p-value grid dataframe.
        alpha (float, optional): Significance level.

    Returns:
        pd.DataFrame: One-row dataframe with lower and upper CI bounds.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha should be in (0, 1); got {alpha}.")
    series = p_values.iloc[:, 0]
    accepted = series[series >= alpha]
    if accepted.empty:
        return pd.DataFrame(
            {
                f"{int(100 * (1 - alpha))}_ci_lower": [np.nan],
                f"{int(100 * (1 - alpha))}_ci_upper": [np.nan],
            },
            index=[p_values.columns[0]],
        )
    return pd.DataFrame(
        {
            f"{int(100 * (1 - alpha))}_ci_lower": [float(accepted.index.min())],
            f"{int(100 * (1 - alpha))}_ci_upper": [float(accepted.index.max())],
        },
        index=[p_values.columns[0]],
    )


def conformal_confidence_interval(
    panel_data: pd.DataFrame,
    unit_column: str,
    date_column: str,
    outcome_column: str,
    treated_unit: Union[str, int],
    intervention_start: DateLike,
    window: int,
    nulls: Iterable[float],
    model_kwargs: dict,
    alpha: float = 0.1,
    q: int = 1,
) -> pd.DataFrame:
    """Compute conformal confidence intervals for post-treatment periods.

    Args:
        panel_data (pd.DataFrame): Long-format panel dataset.
        unit_column (str): Unit identifier column.
        date_column (str): Time column.
        outcome_column (str): Outcome variable.
        treated_unit (Union[str, int]): Treated unit identifier.
        intervention_start (DateLike): Intervention start date.
        window (int): Number of post-treatment periods.
        nulls (Iterable[float]): Candidate null effects.
        model_kwargs (dict): Arguments for SynthControlModel.
        alpha (float, optional): Significance level.
        q (int, optional): Norm power parameter.

    Returns:
        pd.DataFrame: Confidence interval bounds by post-treatment period.
    """
    if window <= 0:
        raise ValueError(f"window should be positive; got {window}.")
    data = panel_data.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    intervention_start = pd.to_datetime(intervention_start)
    post_dates = sorted(data.loc[data[date_column] >= intervention_start, date_column].unique())
    periods = post_dates[:window]
    if len(periods) < window:
        raise ValueError(
            f"Requested window={window}, but only {len(periods)} post-intervention periods are available."
        )

    period_cis = []
    for period in periods:
        p_vals = conformal_p_value_grid_for_period(
            panel_data=data,
            unit_column=unit_column,
            date_column=date_column,
            outcome_column=outcome_column,
            treated_unit=treated_unit,
            intervention_start=intervention_start,
            period=period,
            nulls=nulls,
            model_kwargs=model_kwargs,
            q=q,
        )
        period_cis.append(
            conformal_confidence_interval_from_p_values(p_vals, alpha=alpha)
        )
    return pd.concat(period_cis)
