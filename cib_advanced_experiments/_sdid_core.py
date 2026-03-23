"""Core computational helpers for Synthetic Difference-in-Differences."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cib_advanced_experiments._did_utils import validate_binary_column
from cib_advanced_experiments.model import SynthControlModel


@dataclass(frozen=True)
class _SdidInferenceSummary:
    """Internal container for weighted-regression inference outputs.

    Attributes:
        estimate (float): SDID estimate from weighted regression.
        std_error (float): Standard error of estimate.
        t_statistic (float): t-statistic for estimate.
        pvalue (float): Two-sided p-value.
        ci_lower (float): Lower confidence bound.
        ci_upper (float): Upper confidence bound.
        inference_df (int): Degrees of freedom used for inference.
        used_clustered_se (bool): Whether cluster-robust covariance was used.
        cluster_column (str | None): Cluster column used in inference.
    """

    estimate: float
    std_error: float
    t_statistic: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    inference_df: int
    used_clustered_se: bool
    cluster_column: str | None


@dataclass(frozen=True)
class _SdidCellMetrics:
    """Internal container with 2x2 weighted means and derived metrics.

    Attributes:
        group_means (pd.DataFrame): Weighted group/time means in long format.
        counterfactual_treated_post (float): Treated-post prediction under no treatment.
        treated_change (float): Before/after change for treated group.
        control_change (float): Before/after change for control group.
        naive_before_after (float): Naive treated before/after estimate.
        naive_post_period_comparison (float): Naive treated-vs-control in post period.
    """

    group_means: pd.DataFrame
    counterfactual_treated_post: float
    treated_change: float
    control_change: float
    naive_before_after: float
    naive_post_period_comparison: float


def _prepare_sdid_panel(
    data: pd.DataFrame,
    outcome_column: str,
    unit_column: str,
    time_column: str,
    treatment_column: str,
    post_column: str,
) -> pd.DataFrame:
    """Validate and normalize panel to one row per unit-time cell.

    Args:
        data (pd.DataFrame): Input panel dataframe.
        outcome_column (str): Outcome column name.
        unit_column (str): Unit identifier column.
        time_column (str): Time column.
        treatment_column (str): Binary treatment indicator column.
        post_column (str): Binary post-period indicator column.

    Returns:
        pd.DataFrame: Prepared panel with one row per unit-time cell.
    """
    required_columns = [
        outcome_column,
        unit_column,
        time_column,
        treatment_column,
        post_column,
    ]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    prepared = data.copy().dropna(subset=required_columns)
    if prepared.empty:
        raise ValueError("No rows left after dropping NaN in required columns.")

    if not pd.api.types.is_numeric_dtype(prepared[outcome_column]):
        raise ValueError(f"Column {outcome_column} should be numeric.")

    prepared[unit_column] = prepared[unit_column].astype(str)
    prepared[treatment_column] = prepared[treatment_column].astype(int)
    prepared[post_column] = prepared[post_column].astype(int)

    validate_binary_column(prepared, treatment_column)
    validate_binary_column(prepared, post_column)

    cell_uniques = prepared.groupby([unit_column, time_column], as_index=False)[
        [treatment_column, post_column]
    ].nunique()
    inconsistent_cells = cell_uniques[
        (cell_uniques[treatment_column] > 1) | (cell_uniques[post_column] > 1)
    ]
    if not inconsistent_cells.empty:
        raise ValueError(
            "Columns treatment/post should be constant within each unit-time cell."
        )

    passthrough_columns = [
        column for column in prepared.columns if column not in required_columns
    ]
    aggregations: dict[str, str] = {
        outcome_column: "mean",
        treatment_column: "first",
        post_column: "first",
    }
    for column in passthrough_columns:
        aggregations[column] = "first"

    return (
        prepared.groupby([unit_column, time_column], as_index=False)
        .agg(aggregations)
        .sort_values([time_column, unit_column])
        .reset_index(drop=True)
    )


def _build_synth_control_initial_weights(
    design_matrix: np.ndarray,
    target: np.ndarray,
    max_iter: int,
    tolerance: float,
) -> np.ndarray:
    """Construct simplex warm-start weights using SynthControlModel.

    Args:
        design_matrix (np.ndarray): Design matrix with shape (n_samples, n_features).
        target (np.ndarray): Target vector with shape (n_samples,).
        max_iter (int): Optimization iteration budget.
        tolerance (float): Optimization tolerance.

    Returns:
        np.ndarray: Feasible simplex weights used as warm start.
    """
    n_features = int(design_matrix.shape[1])
    uniform_weights = np.full(shape=n_features, fill_value=1.0 / n_features)

    try:
        centered_X = design_matrix - design_matrix.mean(axis=0, keepdims=True)
        centered_y = target - target.mean()
        warm_start_model = SynthControlModel(
            seed=42,
            loss_func="mse",
            epochs=max(500, int(max_iter)),
            lr=max(1e-12, float(tolerance)),
        )
        warm_start_model.fit(X=centered_X, y=centered_y)
        candidate_weights = np.asarray(warm_start_model.get_weights(), dtype=float)
    except Exception:
        candidate_weights = uniform_weights

    candidate_weights = np.clip(candidate_weights.reshape(-1), 0.0, 1.0)
    total_weight = float(candidate_weights.sum())
    if np.isclose(total_weight, 0.0):
        return uniform_weights
    return candidate_weights / total_weight


def _solve_simplex_with_intercept(
    design_matrix: np.ndarray,
    target: np.ndarray,
    ridge_penalty: float,
    max_iter: int,
    tolerance: float,
) -> tuple[float, np.ndarray]:
    """Solve constrained ridge problem with simplex weights and free intercept.

    Objective:
    min ||intercept + Xw - y||^2 + ridge*||w||^2
    s.t. w_i >= 0 and sum_i w_i = 1.

    Args:
        design_matrix (np.ndarray): Design matrix X with shape (n_samples, n_features).
        target (np.ndarray): Target vector y with shape (n_samples,).
        ridge_penalty (float): Ridge penalty coefficient.
        max_iter (int): Maximum SLSQP iterations.
        tolerance (float): Function tolerance for SLSQP termination.

    Returns:
        tuple[float, np.ndarray]: Estimated intercept and simplex weight vector.
    """
    X = np.asarray(design_matrix, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"design_matrix should be 2D, got shape={X.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "design_matrix and target should have same row count, "
            f"got {X.shape[0]} and {y.shape[0]}."
        )
    if X.shape[1] == 0:
        raise ValueError("design_matrix should have at least one donor column.")

    ridge_penalty = max(0.0, float(ridge_penalty))
    n_features = int(X.shape[1])
    initial_weights = _build_synth_control_initial_weights(
        design_matrix=X,
        target=y,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    initial_intercept = float(np.mean(y - X @ initial_weights))
    initial_guess = np.concatenate([[initial_intercept], initial_weights])

    def objective(params: np.ndarray) -> float:
        intercept = float(params[0])
        weights = params[1:]
        residuals = intercept + X @ weights - y
        return float(
            np.sum(np.square(residuals)) + ridge_penalty * np.sum(np.square(weights))
        )

    constraints = [{"type": "eq", "fun": lambda params: np.sum(params[1:]) - 1.0}]
    bounds = [(None, None)] + [(0.0, 1.0) for _ in range(n_features)]

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": int(max_iter), "ftol": float(tolerance)},
    )

    if not result.success:
        fallback_weights = np.full(shape=n_features, fill_value=1.0 / n_features)
        fallback_intercept = float(np.mean(y - X @ fallback_weights))
        fallback_guess = np.concatenate([[fallback_intercept], fallback_weights])
        result = minimize(
            objective,
            fallback_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": int(max_iter), "ftol": float(tolerance)},
        )
    if not result.success:
        raise ValueError(f"SLSQP optimization failed: {result.message}")

    intercept = float(result.x[0])
    weights = np.asarray(result.x[1:], dtype=float)
    weights = np.clip(weights, 0.0, 1.0)
    weight_sum = float(weights.sum())
    if np.isclose(weight_sum, 0.0):
        raise ValueError("SLSQP produced near-zero weights sum.")
    weights = weights / weight_sum
    return intercept, weights


def _calculate_first_difference_std(
    data: pd.DataFrame,
    outcome_column: str,
    unit_column: str,
    time_column: str,
    treatment_column: str,
    post_column: str,
) -> float:
    """Estimate std of first differences on control pre-treatment rows.

    Args:
        data (pd.DataFrame): Input panel dataframe.
        outcome_column (str): Outcome column name.
        unit_column (str): Unit identifier column.
        time_column (str): Time column.
        treatment_column (str): Treatment indicator column.
        post_column (str): Post-period indicator column.

    Returns:
        float: Standard deviation of first differences with epsilon fallback.
    """
    control_pre = data[(data[treatment_column] == 0) & (data[post_column] == 0)].sort_values(
        [unit_column, time_column]
    )
    first_diff_std = (
        control_pre.groupby(unit_column)[outcome_column].diff().astype(float).std()
    )
    if pd.isna(first_diff_std) or np.isclose(float(first_diff_std), 0.0):
        return 1e-8
    return float(first_diff_std)


def calculate_sigma_hat(
    panel: pd.DataFrame,
    outcome_col: str,
    unit_column: str = "unit",
    time_column: str = "time",
    treatment_column: str = "treated",
    post_column: str = "post",
) -> float:
    """Estimate sigma of first differences on control pre-period observations.

    Args:
        panel (pd.DataFrame): Input panel dataframe.
        outcome_col (str): Outcome column.
        unit_column (str, optional): Unit identifier column.
        time_column (str, optional): Time column.
        treatment_column (str, optional): Treatment indicator column.
        post_column (str, optional): Post-period indicator column.

    Returns:
        float: Estimated sigma-hat value.
    """
    prepared_panel = _prepare_sdid_panel(
        data=panel,
        outcome_column=outcome_col,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
    )
    return _calculate_first_difference_std(
        data=prepared_panel,
        outcome_column=outcome_col,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
    )


def calculate_time_regularization(
    panel: pd.DataFrame,
    outcome_col: str,
    unit_column: str = "unit",
    time_column: str = "time",
    treatment_column: str = "treated",
    post_column: str = "post",
    ridge_scale: float = 1e-6,
) -> float:
    """Calculate SDID time-weight regularization coefficient.

    Args:
        panel (pd.DataFrame): Input panel dataframe.
        outcome_col (str): Outcome column.
        unit_column (str, optional): Unit identifier column.
        time_column (str, optional): Time column.
        treatment_column (str, optional): Treatment indicator column.
        post_column (str, optional): Post-period indicator column.
        ridge_scale (float, optional): Multiplicative scale for sigma-hat.

    Returns:
        float: Time regularization value.
    """
    if ridge_scale <= 0:
        return 0.0
    sigma_hat = calculate_sigma_hat(
        panel=panel,
        outcome_col=outcome_col,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
    )
    return float(max(0.0, float(ridge_scale)) * sigma_hat)


def calculate_regularization(
    panel: pd.DataFrame,
    outcome_col: str,
    unit_column: str = "unit",
    time_column: str = "time",
    treatment_column: str = "treated",
    post_column: str = "post",
) -> float:
    """Calculate SDID unit-weight regularization zeta.

    Args:
        panel (pd.DataFrame): Input panel dataframe.
        outcome_col (str): Outcome column.
        unit_column (str, optional): Unit identifier column.
        time_column (str, optional): Time column.
        treatment_column (str, optional): Treatment indicator column.
        post_column (str, optional): Post-period indicator column.

    Returns:
        float: Unit regularization value zeta.
    """
    prepared_panel = _prepare_sdid_panel(
        data=panel,
        outcome_column=outcome_col,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
    )

    n_treated_post = int(
        (
            (prepared_panel[treatment_column].astype(int) == 1)
            & (prepared_panel[post_column].astype(int) == 1)
        ).sum()
    )
    if n_treated_post <= 0:
        raise ValueError("Need treated-post observations to calculate regularization.")

    sigma_hat = _calculate_first_difference_std(
        data=prepared_panel,
        outcome_column=outcome_col,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
    )
    return float((n_treated_post**0.25) * sigma_hat)


def _fit_unit_weights_core(
    panel: pd.DataFrame,
    outcome_column: str,
    unit_column: str,
    time_column: str,
    treatment_column: str,
    post_column: str,
    optimizer_max_iter: int,
    optimizer_tolerance: float,
) -> tuple[pd.Series, float]:
    """Estimate SDID unit weights on pre-treatment data.

    Args:
        panel (pd.DataFrame): Prepared panel dataframe.
        outcome_column (str): Outcome column used for fitting.
        unit_column (str): Unit identifier column.
        time_column (str): Time column.
        treatment_column (str): Treatment indicator column.
        post_column (str): Post-period indicator column.
        optimizer_max_iter (int): Maximum SLSQP iterations.
        optimizer_tolerance (float): Function tolerance for SLSQP.

    Returns:
        tuple[pd.Series, float]: Unit weights and unit regularization value.
    """
    pre_data = panel[panel[post_column] == 0].copy()
    control_pre = pre_data[pre_data[treatment_column] == 0]
    treated_pre = pre_data[pre_data[treatment_column] == 1]

    if control_pre.empty:
        raise ValueError("No control units in pre-period. Cannot fit unit weights.")
    if treated_pre.empty:
        raise ValueError("No treated units in pre-period. Cannot fit unit weights.")

    y_pre_control = (
        control_pre.pivot(index=time_column, columns=unit_column, values=outcome_column)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    valid_control_columns = y_pre_control.columns[y_pre_control.notna().any(axis=0)]
    y_pre_control = y_pre_control.loc[:, valid_control_columns]

    if y_pre_control.empty:
        raise ValueError("No control units with observed pre-period outcome.")

    if y_pre_control.isna().sum().sum() > 0:
        y_pre_control = y_pre_control.apply(
            lambda series: series.fillna(series.mean() if pd.notna(series.mean()) else 0.0)
        )

    y_pre_treated_mean = (
        treated_pre.groupby(time_column)[outcome_column].mean().reindex(y_pre_control.index)
    )
    if y_pre_treated_mean.isna().sum() > 0:
        mean_value = y_pre_treated_mean.mean()
        y_pre_treated_mean = y_pre_treated_mean.fillna(
            mean_value if pd.notna(mean_value) else 0.0
        )

    if not np.isfinite(y_pre_control.to_numpy()).all():
        raise ValueError("NaN/inf in control pre-treatment matrix after processing.")
    if not np.isfinite(y_pre_treated_mean.to_numpy()).all():
        raise ValueError("NaN/inf in treated pre-treatment mean after processing.")

    zeta = calculate_regularization(
        panel=panel,
        outcome_col=outcome_column,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
    )
    ridge_penalty = float(y_pre_control.shape[0] * (zeta**2))

    _, unit_weights = _solve_simplex_with_intercept(
        design_matrix=y_pre_control.to_numpy(dtype=float),
        target=y_pre_treated_mean.to_numpy(dtype=float),
        ridge_penalty=ridge_penalty,
        max_iter=optimizer_max_iter,
        tolerance=optimizer_tolerance,
    )
    return (
        pd.Series(unit_weights, index=y_pre_control.columns, name="unit_weight"),
        zeta,
    )


def _fit_time_weights_core(
    panel: pd.DataFrame,
    outcome_column: str,
    unit_column: str,
    time_column: str,
    treatment_column: str,
    post_column: str,
    time_weight_ridge_scale: float,
    optimizer_max_iter: int,
    optimizer_tolerance: float,
) -> tuple[pd.Series, float]:
    """Estimate SDID time weights on control series.

    Args:
        panel (pd.DataFrame): Prepared panel dataframe.
        outcome_column (str): Outcome column used for fitting.
        unit_column (str): Unit identifier column.
        time_column (str): Time column.
        treatment_column (str): Treatment indicator column.
        post_column (str): Post-period indicator column.
        time_weight_ridge_scale (float): Time regularization scale.
        optimizer_max_iter (int): Maximum SLSQP iterations.
        optimizer_tolerance (float): Function tolerance for SLSQP.

    Returns:
        tuple[pd.Series, float]: Time weights and time regularization value.
    """
    control = panel[panel[treatment_column] == 0].copy()
    control_pre = control[control[post_column] == 0]
    control_post = control[control[post_column] == 1]

    if control_pre.empty:
        raise ValueError("No control observations in pre-period for time weights.")
    if control_post.empty:
        raise ValueError("No control observations in post-period for time weights.")

    y_pre = (
        control_pre.pivot(index=time_column, columns=unit_column, values=outcome_column)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    y_post_mean = control_post.groupby(unit_column)[outcome_column].mean().reindex(y_pre.columns)

    valid_columns = y_post_mean[y_post_mean.notna()].index
    y_pre = y_pre.loc[:, valid_columns]
    y_post_mean = y_post_mean.loc[valid_columns]

    if y_pre.empty:
        raise ValueError("No control units with observed post-period outcome.")

    if y_pre.isna().sum().sum() > 0:
        y_pre = y_pre.apply(
            lambda series: series.fillna(series.mean() if pd.notna(series.mean()) else 0.0)
        )

    if not np.isfinite(y_pre.to_numpy()).all():
        raise ValueError("NaN/inf in control pre matrix after processing.")
    if not np.isfinite(y_post_mean.to_numpy()).all():
        raise ValueError("NaN/inf in control post mean after processing.")

    time_regularization = calculate_time_regularization(
        panel=panel,
        outcome_col=outcome_column,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
        ridge_scale=time_weight_ridge_scale,
    )
    ridge_penalty = float(y_pre.shape[1] * (time_regularization**2))

    _, time_weights = _solve_simplex_with_intercept(
        design_matrix=y_pre.to_numpy(dtype=float).T,
        target=y_post_mean.to_numpy(dtype=float),
        ridge_penalty=ridge_penalty,
        max_iter=optimizer_max_iter,
        tolerance=optimizer_tolerance,
    )
    return (
        pd.Series(time_weights, index=y_pre.index, name="time_weight"),
        time_regularization,
    )


def _join_sdid_weights_core(
    panel: pd.DataFrame,
    unit_weights: pd.Series,
    time_weights: pd.Series,
    unit_column: str,
    time_column: str,
    treatment_column: str,
    post_column: str,
) -> pd.DataFrame:
    """Merge unit/time weights into panel and build final SDID weights.

    Args:
        panel (pd.DataFrame): Prepared panel dataframe.
        unit_weights (pd.Series): Unit weights indexed by unit.
        time_weights (pd.Series): Time weights indexed by pre-period timestamps.
        unit_column (str): Unit identifier column.
        time_column (str): Time column.
        treatment_column (str): Treatment indicator column.
        post_column (str): Post-period indicator column.

    Returns:
        pd.DataFrame: Panel with unit/time/sdid weights columns.
    """
    weighted = panel.copy()
    n_treated_units = int(
        weighted.loc[weighted[treatment_column] == 1, unit_column].nunique()
    )
    n_post_periods = int(weighted.loc[weighted[post_column] == 1, time_column].nunique())
    if n_treated_units == 0:
        raise ValueError("No treated units available to fill treated weights.")
    if n_post_periods == 0:
        raise ValueError("No post periods available to fill post time weights.")

    weighted = weighted.merge(
        unit_weights.rename("unit_weight"),
        left_on=unit_column,
        right_index=True,
        how="left",
    )
    weighted = weighted.merge(
        time_weights.rename("time_weight"),
        left_on=time_column,
        right_index=True,
        how="left",
    )

    weighted["unit_weight"] = weighted["unit_weight"].fillna(1.0 / n_treated_units)
    weighted["time_weight"] = weighted["time_weight"].fillna(1.0 / n_post_periods)
    weighted["unit_weight"] = weighted["unit_weight"].astype(float).clip(lower=0.0)
    weighted["time_weight"] = weighted["time_weight"].astype(float).clip(lower=0.0)
    weighted["sdid_weight"] = (weighted["unit_weight"] * weighted["time_weight"]).clip(
        lower=1e-12
    )
    return weighted


def _cluster_robust_covariance(
    X: np.ndarray,
    residuals: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Compute one-way cluster-robust covariance for weighted least squares.

    Args:
        X (np.ndarray): Design matrix with shape (n_obs, n_params).
        residuals (np.ndarray): Residual vector with shape (n_obs,).
        weights (np.ndarray): Observation weights with shape (n_obs,).
        clusters (np.ndarray): Cluster labels with shape (n_obs,).

    Returns:
        tuple[np.ndarray, int]: Covariance matrix and number of unique clusters.
    """
    if X.ndim != 2:
        raise ValueError(f"X should be 2D array, got shape={X.shape}.")
    if residuals.ndim != 1:
        raise ValueError(f"residuals should be 1D array, got shape={residuals.shape}.")
    if weights.ndim != 1:
        raise ValueError(f"weights should be 1D array, got shape={weights.shape}.")
    if clusters.ndim != 1:
        raise ValueError(f"clusters should be 1D array, got shape={clusters.shape}.")

    n_obs, n_params = X.shape
    if not (
        residuals.shape[0] == n_obs
        and weights.shape[0] == n_obs
        and clusters.shape[0] == n_obs
    ):
        raise ValueError("X, residuals, weights and clusters should have same row count.")

    xtwx = X.T @ (X * weights[:, None])
    bread = np.linalg.pinv(xtwx)

    cluster_labels = pd.Series(clusters).astype(str).to_numpy()
    unique_clusters = pd.Series(cluster_labels).unique().tolist()
    n_clusters = len(unique_clusters)
    if n_clusters == 0:
        raise ValueError("No clusters provided for cluster-robust covariance.")

    meat = np.zeros((n_params, n_params), dtype=float)
    for cluster in unique_clusters:
        mask = cluster_labels == str(cluster)
        X_g = X[mask]
        u_g = residuals[mask]
        w_g = weights[mask]
        score_g = X_g.T @ (w_g * u_g)
        meat += np.outer(score_g, score_g)

    covariance = bread @ meat @ bread
    if n_clusters > 1 and n_obs > n_params:
        correction = (n_clusters / (n_clusters - 1.0)) * (
            (n_obs - 1.0) / (n_obs - n_params)
        )
        covariance = correction * covariance
    return covariance, n_clusters


__all__ = [
    "_SdidCellMetrics",
    "_SdidInferenceSummary",
    "_cluster_robust_covariance",
    "_fit_time_weights_core",
    "_fit_unit_weights_core",
    "_join_sdid_weights_core",
    "_prepare_sdid_panel",
    "calculate_regularization",
    "calculate_sigma_hat",
    "calculate_time_regularization",
]
