"""High-level Synthetic Difference-in-Differences APIs.

The implementation builds on the SDID presentation in Matheus Facure's
*Causal Inference for the Brave and True* and the estimator of Arkhangelsky
et al. (2021). See ``THIRD_PARTY_NOTICES.md`` for attribution and licensing.
"""

from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from scipy import stats

from causal_loom._did_utils import (
    safe_weighted_mean as _safe_weighted_mean,
)
from causal_loom._sdid_core import (
    _cluster_robust_covariance,
    _fit_time_weights_core,
    _fit_unit_weights_core,
    _join_sdid_weights_core,
    _prepare_sdid_panel,
    _SdidCellMetrics,
    _SdidInferenceSummary,
    calculate_sigma_hat,
)
from causal_loom._sdid_core import (
    calculate_regularization as calculate_regularization,
)
from causal_loom._sdid_core import (
    calculate_time_regularization as calculate_time_regularization,
)
from causal_loom.visualization import (
    create_did_2x2_figure,
    create_did_parallel_trends_figure,
)


def residualize_outcome(
    panel: pd.DataFrame,
    outcome_col: str,
    covariate_cols: list[str],
    treatment_column: str = "treated",
    post_column: str = "post",
    residual_col: str | None = None,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Residualize outcome by covariates fitted on untreated observations.

    Fitting mask matches sdid.py logic:
    - all control rows, and
    - treated rows in pre-period.

    Args:
        panel (pd.DataFrame): Input panel dataframe.
        outcome_col (str): Outcome column to residualize.
        covariate_cols (list[str]): Covariate columns for residualization model.
        treatment_column (str, optional): Treatment indicator column.
        post_column (str, optional): Post-period indicator column.
        residual_col (str | None, optional): Output residual column name.

    Returns:
        tuple[pd.DataFrame, str, dict[str, Any]]: Residualized dataframe, residual
            column name, and residualization model metadata.
    """
    if outcome_col not in panel.columns:
        raise ValueError(f"Column '{outcome_col}' is not present in panel.")
    if treatment_column not in panel.columns:
        raise ValueError(f"Column '{treatment_column}' is not present in panel.")
    if post_column not in panel.columns:
        raise ValueError(f"Column '{post_column}' is not present in panel.")
    if not covariate_cols:
        raise ValueError("covariate_cols must be non-empty for residualization.")
    missing = [column for column in covariate_cols if column not in panel.columns]
    if missing:
        raise ValueError(f"Missing covariate columns: {missing}")

    out = panel.copy()
    residual_col = residual_col or f"{outcome_col}_resid"
    X = out[covariate_cols].copy()

    for column in X.columns:
        if pd.api.types.is_bool_dtype(X[column]):
            X[column] = X[column].astype(int)

    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numeric_columns = [
        column for column in X.columns if column not in categorical_columns
    ]

    for column in numeric_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")
        fill_value = X[column].median()
        X[column] = X[column].fillna(fill_value if pd.notna(fill_value) else 0.0)
    for column in categorical_columns:
        X[column] = X[column].astype("string").fillna("__missing__")

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True,
        dtype=float,
    )
    constant_columns = [
        column for column in X.columns if X[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        X = X.drop(columns=constant_columns)

    X.insert(0, "_intercept", 1.0)
    fit_mask = (out[treatment_column].astype(int) == 0) | (
        out[post_column].astype(int) == 0
    )
    X_fit = X.loc[fit_mask].to_numpy(dtype=float)
    y_fit = out.loc[fit_mask, outcome_col].astype(float).to_numpy()
    if X_fit.shape[0] <= X_fit.shape[1]:
        raise ValueError(
            "Not enough untreated observations to residualize outcome. "
            f"Need more than {X_fit.shape[1]} rows, got {X_fit.shape[0]}."
        )

    coefficients, _, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    predictions = X.to_numpy(dtype=float) @ coefficients
    out[residual_col] = out[outcome_col].astype(float) - predictions
    if not np.isfinite(out[residual_col].to_numpy()).all():
        raise ValueError("Residualization produced NaN/inf values.")

    model_info = {
        "coefficients": pd.Series(coefficients, index=X.columns, name="coef"),
        "feature_columns": X.columns.tolist(),
        "n_fit": int(X_fit.shape[0]),
    }
    return out, residual_col, model_info


@dataclass(frozen=True)
class SyntheticDifferenceInDifferencesResult:
    """Container with SDID estimate, weights and inference outputs.

    Attributes:
        estimate (float): SDID estimate (interaction coefficient).
        std_error (float): Standard error of SDID estimate from weighted OLS.
        t_statistic (float): t-statistic for SDID estimate.
        pvalue (float): Two-sided p-value for SDID estimate.
        ci_lower (float): Lower confidence bound.
        ci_upper (float): Upper confidence bound.
        alpha (float): Significance level used for confidence interval.
        n_obs (int): Number of observations used in estimation.
        unit_weights (pd.Series): Estimated unit weights for control units.
        time_weights (pd.Series): Estimated time weights for pre-treatment periods.
        weighted_data (pd.DataFrame): Panel with merged unit/time/combined SDID weights.
        group_means (pd.DataFrame): Weighted mean outcome by treatment/post cells.
        counterfactual_treated_post (float): Predicted treated-post mean under no treatment.
        treated_change (float): Before/after change in treated group.
        control_change (float): Before/after change in control group.
        naive_before_after (float): Naive before/after estimate (treated only).
        naive_post_period_comparison (float): Naive post-period treated-vs-control estimate.
        unit_regularization (float): Unit-weight regularization coefficient zeta.
        time_regularization (float): Time-weight regularization coefficient.
        model_outcome_column (str): Outcome column used in regression (can be residualized).
        residualized (bool): Whether covariate residualization was used.
        covariate_columns (list[str]): Covariates used for residualization.
        residualization_model (dict[str, Any] | None): Residualization model metadata.
        used_fixed_effects (bool): Whether unit/time fixed effects were used.
        used_clustered_se (bool): Whether clustered standard errors were used.
        cluster_column (str | None): Cluster column used in inference.
        inference_df (int): Degrees of freedom used in t-based inference.
    """

    estimate: float
    std_error: float
    t_statistic: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    alpha: float
    n_obs: int
    unit_weights: pd.Series
    time_weights: pd.Series
    weighted_data: pd.DataFrame
    group_means: pd.DataFrame
    counterfactual_treated_post: float
    treated_change: float
    control_change: float
    naive_before_after: float
    naive_post_period_comparison: float
    unit_regularization: float
    time_regularization: float
    model_outcome_column: str
    residualized: bool
    covariate_columns: list[str]
    residualization_model: dict[str, Any] | None
    used_fixed_effects: bool
    used_clustered_se: bool
    cluster_column: str | None
    inference_df: int


@dataclass(frozen=True)
class StaggeredSyntheticDifferenceInDifferencesResult:
    """Container with cohort-level SDID estimates for staggered adoption.

    Attributes:
        aggregate_estimate (float): Weighted average ATT across cohorts.
        cohort_results (pd.DataFrame): Cohort-level SDID estimates and diagnostics.
        cohort_models (dict[Any, SyntheticDifferenceInDifferencesResult]): Fitted cohort models.
        weighting_column (str): Cohort-results column used for aggregate weighting.
    """

    aggregate_estimate: float
    cohort_results: pd.DataFrame
    cohort_models: dict[Any, SyntheticDifferenceInDifferencesResult]
    weighting_column: str


class SyntheticDifferenceInDifferences:
    """Synthetic Difference-in-Differences estimator for block treatment assignment.

    Inspired by:
    https://matheusfacure.github.io/python-causality-handbook/25-Synthetic-Diff-in-Diff.html

    Args:
        data (pd.DataFrame): Input panel dataframe.
        outcome_column (str): Numeric outcome column.
        unit_column (str): Unit identifier column.
        time_column (str): Time column.
        treatment_column (str): Binary treatment-group indicator (1 treated, 0 control).
        post_column (str): Binary post-period indicator (1 post, 0 pre).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome_column: str,
        unit_column: str,
        time_column: str,
        treatment_column: str,
        post_column: str,
    ) -> None:
        """Initialize SDID estimator and normalize inputs.

        Args:
            data (pd.DataFrame): Input panel dataframe.
            outcome_column (str): Numeric outcome column.
            unit_column (str): Unit identifier column.
            time_column (str): Time column.
            treatment_column (str): Binary treatment indicator column.
            post_column (str): Binary post-period indicator column.

        Returns:
            None
        """
        self.data = data.copy()
        self.outcome_column = outcome_column
        self.unit_column = unit_column
        self.time_column = time_column
        self.treatment_column = treatment_column
        self.post_column = post_column
        self.latest_result: SyntheticDifferenceInDifferencesResult | None = None

        self.__validate_inputs()
        self.__normalize_inputs()

    def __validate_inputs(self) -> None:
        """Validate that required columns exist in input dataframe.

        Args:
            None

        Returns:
            None
        """
        required = [
            self.outcome_column,
            self.unit_column,
            self.time_column,
            self.treatment_column,
            self.post_column,
        ]
        missing = [column for column in required if column not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def __normalize_inputs(self) -> None:
        """Normalize dtypes and aggregate panel to one row per unit-time cell.

        Args:
            None

        Returns:
            None
        """
        self.data = _prepare_sdid_panel(
            data=self.data,
            outcome_column=self.outcome_column,
            unit_column=self.unit_column,
            time_column=self.time_column,
            treatment_column=self.treatment_column,
            post_column=self.post_column,
        )

    def __fit_unit_weights(
        self,
        panel: pd.DataFrame,
        outcome_column: str,
        optimizer_max_iter: int,
        optimizer_tolerance: float,
    ) -> tuple[pd.Series, float]:
        """Estimate SDID unit weights on pre-treatment data.

        Args:
            panel (pd.DataFrame): Prepared panel dataframe.
            outcome_column (str): Outcome column used for fitting.
            optimizer_max_iter (int): Maximum SLSQP iterations.
            optimizer_tolerance (float): Function tolerance for SLSQP.

        Returns:
            tuple[pd.Series, float]: Unit weights and unit regularization value.
        """
        return _fit_unit_weights_core(
            panel=panel,
            outcome_column=outcome_column,
            unit_column=self.unit_column,
            time_column=self.time_column,
            treatment_column=self.treatment_column,
            post_column=self.post_column,
            optimizer_max_iter=optimizer_max_iter,
            optimizer_tolerance=optimizer_tolerance,
        )

    def __fit_time_weights(
        self,
        panel: pd.DataFrame,
        outcome_column: str,
        time_weight_ridge_scale: float,
        optimizer_max_iter: int,
        optimizer_tolerance: float,
    ) -> tuple[pd.Series, float]:
        """Estimate SDID time weights on control series.

        Args:
            panel (pd.DataFrame): Prepared panel dataframe.
            outcome_column (str): Outcome column used for fitting.
            time_weight_ridge_scale (float): Time regularization scale.
            optimizer_max_iter (int): Maximum SLSQP iterations.
            optimizer_tolerance (float): Function tolerance for SLSQP.

        Returns:
            tuple[pd.Series, float]: Time weights and time regularization value.
        """
        return _fit_time_weights_core(
            panel=panel,
            outcome_column=outcome_column,
            unit_column=self.unit_column,
            time_column=self.time_column,
            treatment_column=self.treatment_column,
            post_column=self.post_column,
            time_weight_ridge_scale=time_weight_ridge_scale,
            optimizer_max_iter=optimizer_max_iter,
            optimizer_tolerance=optimizer_tolerance,
        )

    def __join_weights(
        self,
        panel: pd.DataFrame,
        unit_weights: pd.Series,
        time_weights: pd.Series,
    ) -> pd.DataFrame:
        """Merge unit/time weights into panel and build final SDID weights.

        Args:
            panel (pd.DataFrame): Prepared panel dataframe.
            unit_weights (pd.Series): Unit weights indexed by unit.
            time_weights (pd.Series): Time weights indexed by pre-period timestamps.

        Returns:
            pd.DataFrame: Panel with unit/time/sdid weights columns.
        """
        return _join_sdid_weights_core(
            panel=panel,
            unit_weights=unit_weights,
            time_weights=time_weights,
            unit_column=self.unit_column,
            time_column=self.time_column,
            treatment_column=self.treatment_column,
            post_column=self.post_column,
        )

    def __validate_fit_arguments(
        self,
        alpha: float,
        time_weight_ridge_scale: float,
        cluster_column: str | None,
        optimizer_max_iter: int,
        optimizer_tolerance: float,
    ) -> None:
        """Validate user-facing fit arguments before estimation.

        Args:
            alpha (float): Significance level for confidence interval.
            time_weight_ridge_scale (float): Time-weight regularization scale.
            cluster_column (str | None): Optional cluster column.
            optimizer_max_iter (int): Maximum SLSQP iterations.
            optimizer_tolerance (float): SLSQP function tolerance.

        Returns:
            None
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha should be in (0, 1); got {alpha}.")
        if time_weight_ridge_scale < 0:
            raise ValueError(
                "time_weight_ridge_scale should be non-negative, "
                f"got {time_weight_ridge_scale}."
            )
        if optimizer_max_iter <= 0:
            raise ValueError(
                f"optimizer_max_iter should be positive, got {optimizer_max_iter}."
            )
        if optimizer_tolerance <= 0:
            raise ValueError(
                f"optimizer_tolerance should be positive, got {optimizer_tolerance}."
            )
        if cluster_column is not None and cluster_column not in self.data.columns:
            raise ValueError(f"Column '{cluster_column}' not found in data.")

    def __prepare_fit_panel(
        self,
        covariate_columns: list[str] | None,
    ) -> tuple[pd.DataFrame, str, dict[str, Any] | None, list[str]]:
        """Prepare panel and optionally residualize outcome for fit().

        Args:
            covariate_columns (list[str] | None): Optional residualization covariates.

        Returns:
            tuple[pd.DataFrame, str, dict[str, Any] | None, list[str]]: Prepared panel,
                model outcome column, residualization metadata and normalized covariates.
        """
        panel = self.data.copy()
        normalized_covariates = list(covariate_columns or [])
        model_outcome_column = self.outcome_column
        residualization_model: dict[str, Any] | None = None
        if normalized_covariates:
            panel, model_outcome_column, residualization_model = residualize_outcome(
                panel=panel,
                outcome_col=self.outcome_column,
                covariate_cols=normalized_covariates,
                treatment_column=self.treatment_column,
                post_column=self.post_column,
                residual_col=f"{self.outcome_column}_resid",
            )
        return panel, model_outcome_column, residualization_model, normalized_covariates

    def __build_regression_inputs(
        self,
        weighted_data: pd.DataFrame,
        outcome_column: str,
        use_fixed_effects: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Build weighted regression design matrix and vectors.

        Args:
            weighted_data (pd.DataFrame): Panel with SDID weights.
            outcome_column (str): Outcome column used in final regression.
            use_fixed_effects (bool): Whether to include unit/time fixed effects.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, int]: Design matrix X, outcome y,
                weights w and effect-coefficient index.
        """
        d = weighted_data[self.treatment_column].astype(float).to_numpy()
        t = weighted_data[self.post_column].astype(float).to_numpy()
        y = weighted_data[outcome_column].astype(float).to_numpy()
        w = weighted_data["sdid_weight"].astype(float).to_numpy()

        interaction = d * t
        if use_fixed_effects:
            unit_dummies = pd.get_dummies(
                weighted_data[self.unit_column].astype(str),
                drop_first=True,
                dtype=float,
            )
            time_dummies = pd.get_dummies(
                weighted_data[self.time_column],
                drop_first=True,
                dtype=float,
            )
            X_df = pd.concat(
                [
                    pd.Series(np.ones_like(d), name="intercept"),
                    pd.Series(interaction, name="treated_post"),
                    unit_dummies.reset_index(drop=True),
                    time_dummies.reset_index(drop=True),
                ],
                axis=1,
            )
            X = X_df.to_numpy(dtype=float)
            effect_index = 1
        else:
            X = np.column_stack([np.ones_like(d), d, t, interaction])
            effect_index = 3
        return X, y, w, effect_index

    def __estimate_weighted_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        effect_index: int,
        alpha: float,
        cluster_by_unit: bool,
        cluster_column: str | None,
        weighted_data: pd.DataFrame,
    ) -> _SdidInferenceSummary:
        """Estimate weighted regression and build inference summary.

        Args:
            X (np.ndarray): Design matrix.
            y (np.ndarray): Outcome vector.
            w (np.ndarray): Observation weights.
            effect_index (int): Index of treatment effect coefficient in beta.
            alpha (float): Significance level.
            cluster_by_unit (bool): Whether to use clustered standard errors.
            cluster_column (str | None): Optional cluster column.
            weighted_data (pd.DataFrame): Panel with SDID weights and identifiers.

        Returns:
            _SdidInferenceSummary: Weighted-regression inference outputs.
        """
        n_obs, n_params = X.shape
        if n_obs <= n_params:
            raise ValueError(
                f"Need more observations than parameters for inference, got n={n_obs}."
            )

        xtwx = X.T @ (X * w[:, None])
        xtwx_inv = np.linalg.pinv(xtwx)
        beta = xtwx_inv @ (X.T @ (w * y))
        residuals = y - X @ beta
        sigma2 = float((w * np.square(residuals)).sum() / (n_obs - n_params))

        effective_cluster_column: str | None = None
        used_clustered_se = False
        if cluster_by_unit:
            effective_cluster_column = cluster_column or self.unit_column
            clusters = weighted_data[effective_cluster_column].astype(str).to_numpy()
            covariance_candidate, n_clusters = _cluster_robust_covariance(
                X=X,
                residuals=residuals,
                weights=w,
                clusters=clusters,
            )
            if n_clusters > 1:
                cov = covariance_candidate
                inference_df = int(n_clusters - 1)
                used_clustered_se = True
            else:
                cov = sigma2 * xtwx_inv
                inference_df = int(n_obs - n_params)
        else:
            cov = sigma2 * xtwx_inv
            inference_df = int(n_obs - n_params)

        variance = float(cov[effect_index, effect_index])
        std_error = float(np.sqrt(max(0.0, variance)))
        estimate = float(beta[effect_index])
        if np.isclose(std_error, 0.0):
            t_stat = np.nan
            pvalue = np.nan
            ci_delta = np.nan
        else:
            t_stat = float(estimate / std_error)
            pvalue = float(2 * (1 - stats.t.cdf(abs(t_stat), df=inference_df)))
            ci_delta = float(stats.t.ppf(1 - alpha / 2, df=inference_df) * std_error)

        return _SdidInferenceSummary(
            estimate=estimate,
            std_error=std_error,
            t_statistic=t_stat,
            pvalue=pvalue,
            ci_lower=float(estimate - ci_delta),
            ci_upper=float(estimate + ci_delta),
            inference_df=int(inference_df),
            used_clustered_se=used_clustered_se,
            cluster_column=effective_cluster_column,
        )

    def __compute_cell_metrics(
        self,
        weighted_data: pd.DataFrame,
        outcome_column: str,
    ) -> _SdidCellMetrics:
        """Compute weighted 2x2 means and naive/controlled effect decomposition.

        Args:
            weighted_data (pd.DataFrame): Panel with SDID weights.
            outcome_column (str): Outcome column used in final regression.

        Returns:
            _SdidCellMetrics: Weighted group means and derived metrics.
        """
        treated_pre = _safe_weighted_mean(
            weighted_data,
            (weighted_data[self.treatment_column] == 1)
            & (weighted_data[self.post_column] == 0),
            outcome_column,
            "sdid_weight",
        )
        treated_post = _safe_weighted_mean(
            weighted_data,
            (weighted_data[self.treatment_column] == 1)
            & (weighted_data[self.post_column] == 1),
            outcome_column,
            "sdid_weight",
        )
        control_pre = _safe_weighted_mean(
            weighted_data,
            (weighted_data[self.treatment_column] == 0)
            & (weighted_data[self.post_column] == 0),
            outcome_column,
            "sdid_weight",
        )
        control_post = _safe_weighted_mean(
            weighted_data,
            (weighted_data[self.treatment_column] == 0)
            & (weighted_data[self.post_column] == 1),
            outcome_column,
            "sdid_weight",
        )

        treated_change = float(treated_post - treated_pre)
        control_change = float(control_post - control_pre)
        naive_post_period_comparison = float(treated_post - control_post)
        counterfactual_treated_post = float(treated_pre + control_change)
        group_means = pd.DataFrame(
            {
                "group": ["control", "control", "treated", "treated"],
                "post_period": [0, 1, 0, 1],
                "mean_outcome": [control_pre, control_post, treated_pre, treated_post],
            }
        )
        return _SdidCellMetrics(
            group_means=group_means,
            counterfactual_treated_post=counterfactual_treated_post,
            treated_change=treated_change,
            control_change=control_change,
            naive_before_after=treated_change,
            naive_post_period_comparison=naive_post_period_comparison,
        )

    def fit(
        self,
        alpha: float = 0.05,
        time_weight_ridge_scale: float = 1e-6,
        covariate_columns: list[str] | None = None,
        use_fixed_effects: bool = True,
        cluster_by_unit: bool = True,
        cluster_column: str | None = None,
        optimizer_max_iter: int = 5000,
        optimizer_tolerance: float = 1e-9,
    ) -> SyntheticDifferenceInDifferencesResult:
        """Estimate SDID effect, weights and weighted OLS inference.

        Args:
            alpha (float, optional): Significance level for confidence interval.
            time_weight_ridge_scale (float, optional): Scale for time-weight ridge regularization.
            covariate_columns (list[str] | None, optional): Optional covariates for outcome residualization.
            use_fixed_effects (bool, optional): If True, run weighted regression with
                unit/time fixed effects and interaction treated:post.
            cluster_by_unit (bool, optional): If True, compute one-way clustered standard
                errors (cluster by unit by default).
            cluster_column (str | None, optional): Column used for clustering. Defaults to unit column.
            optimizer_max_iter (int, optional): SLSQP max iterations for weight optimization.
            optimizer_tolerance (float, optional): SLSQP function tolerance.

        Returns:
            SyntheticDifferenceInDifferencesResult: Estimated effect and diagnostics.
        """
        self.__validate_fit_arguments(
            alpha=alpha,
            time_weight_ridge_scale=time_weight_ridge_scale,
            cluster_column=cluster_column,
            optimizer_max_iter=optimizer_max_iter,
            optimizer_tolerance=optimizer_tolerance,
        )
        panel, model_outcome_column, residualization_model, normalized_covariates = (
            self.__prepare_fit_panel(covariate_columns=covariate_columns)
        )

        unit_weights, unit_regularization = self.__fit_unit_weights(
            panel=panel,
            outcome_column=model_outcome_column,
            optimizer_max_iter=optimizer_max_iter,
            optimizer_tolerance=optimizer_tolerance,
        )
        time_weights, time_regularization = self.__fit_time_weights(
            panel=panel,
            outcome_column=model_outcome_column,
            time_weight_ridge_scale=time_weight_ridge_scale,
            optimizer_max_iter=optimizer_max_iter,
            optimizer_tolerance=optimizer_tolerance,
        )
        weighted_data = self.__join_weights(
            panel=panel,
            unit_weights=unit_weights,
            time_weights=time_weights,
        )

        X, y, w, effect_index = self.__build_regression_inputs(
            weighted_data=weighted_data,
            outcome_column=model_outcome_column,
            use_fixed_effects=use_fixed_effects,
        )
        inference = self.__estimate_weighted_regression(
            X=X,
            y=y,
            w=w,
            effect_index=effect_index,
            alpha=alpha,
            cluster_by_unit=cluster_by_unit,
            cluster_column=cluster_column,
            weighted_data=weighted_data,
        )
        cell_metrics = self.__compute_cell_metrics(
            weighted_data=weighted_data,
            outcome_column=model_outcome_column,
        )
        n_obs = int(X.shape[0])

        result = SyntheticDifferenceInDifferencesResult(
            estimate=inference.estimate,
            std_error=inference.std_error,
            t_statistic=inference.t_statistic,
            pvalue=inference.pvalue,
            ci_lower=inference.ci_lower,
            ci_upper=inference.ci_upper,
            alpha=alpha,
            n_obs=n_obs,
            unit_weights=unit_weights.sort_values(ascending=False),
            time_weights=time_weights.sort_index(),
            weighted_data=weighted_data,
            group_means=cell_metrics.group_means,
            counterfactual_treated_post=cell_metrics.counterfactual_treated_post,
            treated_change=cell_metrics.treated_change,
            control_change=cell_metrics.control_change,
            naive_before_after=cell_metrics.naive_before_after,
            naive_post_period_comparison=cell_metrics.naive_post_period_comparison,
            unit_regularization=unit_regularization,
            time_regularization=time_regularization,
            model_outcome_column=model_outcome_column,
            residualized=bool(normalized_covariates),
            covariate_columns=normalized_covariates,
            residualization_model=residualization_model,
            used_fixed_effects=use_fixed_effects,
            used_clustered_se=inference.used_clustered_se,
            cluster_column=inference.cluster_column,
            inference_df=inference.inference_df,
        )
        self.latest_result = result
        return result

    def get_estimators_breakdown(self) -> pd.DataFrame:
        """Return naive estimators and SDID decomposition in one table.

        Args:
            None

        Returns:
            pd.DataFrame: Table with estimator names and values.
        """
        if self.latest_result is None:
            raise ValueError("Run fit() before requesting estimators breakdown.")
        result = self.latest_result
        return pd.DataFrame(
            {
                "estimator": [
                    "naive_before_after_treated",
                    "naive_post_period_treated_vs_control",
                    "sdid_controlled_difference",
                ],
                "value": [
                    result.naive_before_after,
                    result.naive_post_period_comparison,
                    result.estimate,
                ],
            }
        )

    def get_parallel_trends_frame(self, pre_period_only: bool = True) -> pd.DataFrame:
        """Prepare weighted treated/control means by time for trend diagnostics.

        Args:
            pre_period_only (bool, optional): Keep only pre-period rows if True.

        Returns:
            pd.DataFrame: Dataframe with columns time, treated_mean, control_mean, gap.
        """
        outcome_column = (
            self.latest_result.model_outcome_column
            if self.latest_result is not None
            else self.outcome_column
        )
        source = (
            self.latest_result.weighted_data.copy()
            if self.latest_result is not None
            else self.data.copy()
        )
        if "unit_weight" not in source.columns:
            source["unit_weight"] = 1.0

        if pre_period_only:
            source = source[source[self.post_column] == 0].copy()
            if source.empty:
                raise ValueError("No pre-period rows available for trend diagnostics.")

        rows = []
        for (time_value, treated_flag), block in source.groupby(
            [self.time_column, self.treatment_column]
        ):
            weights = block["unit_weight"].astype(float).to_numpy()
            values = block[outcome_column].astype(float).to_numpy()
            if np.isclose(float(weights.sum()), 0.0):
                continue
            rows.append(
                {
                    "time": time_value,
                    "treated_flag": int(treated_flag),
                    "mean_outcome": float(np.average(values, weights=weights)),
                }
            )
        grouped = pd.DataFrame(rows)
        if grouped.empty:
            raise ValueError("Trend diagnostics failed: no rows after grouping.")

        pivoted = (
            grouped.pivot(index="time", columns="treated_flag", values="mean_outcome")
            .rename(columns={0: "control_mean", 1: "treated_mean"})
            .reset_index()
            .sort_values("time")
        )
        if (
            "control_mean" not in pivoted.columns
            or "treated_mean" not in pivoted.columns
        ):
            raise ValueError("Trend diagnostics need both treatment groups.")
        pivoted["gap"] = pivoted["treated_mean"] - pivoted["control_mean"]
        return pivoted

    def fit_effects_by_period(
        self,
        alpha: float = 0.05,
        time_weight_ridge_scale: float = 1e-6,
        covariate_columns: list[str] | None = None,
        use_fixed_effects: bool = True,
        cluster_by_unit: bool = True,
        cluster_column: str | None = None,
        optimizer_max_iter: int = 5000,
        optimizer_tolerance: float = 1e-9,
    ) -> pd.DataFrame:
        """Estimate one SDID effect per post-treatment period.

        For each post period t*, runs SDID on rows:
        - all pre-period rows
        - post rows only for time == t*

        Args:
            alpha (float, optional): Significance level for confidence interval.
            time_weight_ridge_scale (float, optional): Time-weight regularization scale.
            covariate_columns (list[str] | None, optional): Optional residualization covariates.
            use_fixed_effects (bool, optional): Use unit/time fixed effects in final regression.
            cluster_by_unit (bool, optional): Use clustered standard errors.
            cluster_column (str | None, optional): Cluster column name.
            optimizer_max_iter (int, optional): Maximum SLSQP iterations.
            optimizer_tolerance (float, optional): Function tolerance for SLSQP.

        Returns:
            pd.DataFrame: Per-period SDID estimates and inference columns.
        """
        post_periods = sorted(
            self.data.loc[self.data[self.post_column] == 1, self.time_column]
            .drop_duplicates()
            .tolist()
        )
        if len(post_periods) == 0:
            raise ValueError("No post periods found to estimate period effects.")

        rows: list[dict[str, Any]] = []
        for period in post_periods:
            period_data = self.data[
                (self.data[self.post_column] == 0)
                | (self.data[self.time_column] == period)
            ].copy()
            # Reuse one-shot API to keep period-level fitting path aligned
            # with the main SDID estimator behavior.
            period_result = synthetic_difference_in_differences(
                data=period_data,
                outcome_column=self.outcome_column,
                unit_column=self.unit_column,
                time_column=self.time_column,
                treatment_column=self.treatment_column,
                post_column=self.post_column,
                alpha=alpha,
                time_weight_ridge_scale=time_weight_ridge_scale,
                covariate_columns=covariate_columns,
                use_fixed_effects=use_fixed_effects,
                cluster_by_unit=cluster_by_unit,
                cluster_column=cluster_column,
                optimizer_max_iter=optimizer_max_iter,
                optimizer_tolerance=optimizer_tolerance,
            )
            rows.append(
                {
                    "period": period,
                    "estimate": period_result.estimate,
                    "std_error": period_result.std_error,
                    "pvalue": period_result.pvalue,
                    "ci_lower": period_result.ci_lower,
                    "ci_upper": period_result.ci_upper,
                    "n_obs": period_result.n_obs,
                }
            )
        return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)

    def plot_2x2(self) -> go.Figure:
        """Plot classic 2x2 SDID chart with counterfactual treated-post point.

        Args:
            None

        Returns:
            go.Figure: Plotly figure.
        """
        if self.latest_result is None:
            raise ValueError("Run fit() before plotting 2x2 SDID chart.")
        fig = create_did_2x2_figure(
            group_means=self.latest_result.group_means,
            counterfactual_treated_post=self.latest_result.counterfactual_treated_post,
            estimate=self.latest_result.estimate,
        )
        fig.show()
        return fig

    def plot_parallel_trends(
        self,
        treatment_start: Union[str, pd.Timestamp, None] = None,
        pre_period_only: bool = False,
    ) -> go.Figure:
        """Plot weighted treated/control trends by time for SDID diagnostics.

        Args:
            treatment_start (Union[str, pd.Timestamp, None], optional): Vertical marker date.
            pre_period_only (bool, optional): Plot only pre-period rows if True.

        Returns:
            go.Figure: Plotly figure.
        """
        trends = self.get_parallel_trends_frame(pre_period_only=pre_period_only)
        outcome_label = (
            self.latest_result.model_outcome_column
            if self.latest_result is not None
            else self.outcome_column
        )
        fig = create_did_parallel_trends_figure(
            trends_frame=trends,
            treatment_start=treatment_start,
            outcome_column=outcome_label,
            pre_period_only=pre_period_only,
        )
        fig.show()
        return fig


def synthetic_difference_in_differences(
    data: pd.DataFrame,
    outcome_column: str,
    unit_column: str,
    time_column: str,
    treatment_column: str,
    post_column: str,
    alpha: float = 0.05,
    time_weight_ridge_scale: float = 1e-6,
    covariate_columns: list[str] | None = None,
    use_fixed_effects: bool = True,
    cluster_by_unit: bool = True,
    cluster_column: str | None = None,
    optimizer_max_iter: int = 5000,
    optimizer_tolerance: float = 1e-9,
) -> SyntheticDifferenceInDifferencesResult:
    """Convenience function for one-shot SDID estimation.

    Args:
        data (pd.DataFrame): Input panel dataframe.
        outcome_column (str): Numeric outcome column.
        unit_column (str): Unit identifier column.
        time_column (str): Time column.
        treatment_column (str): Binary treatment indicator column.
        post_column (str): Binary post-period indicator column.
        alpha (float, optional): Significance level for confidence interval.
        time_weight_ridge_scale (float, optional): Time-weight regularization scale.
        covariate_columns (list[str] | None, optional): Optional residualization covariates.
        use_fixed_effects (bool, optional): Use unit/time fixed effects in final regression.
        cluster_by_unit (bool, optional): Use clustered standard errors.
        cluster_column (str | None, optional): Cluster column name.
        optimizer_max_iter (int, optional): Maximum SLSQP iterations.
        optimizer_tolerance (float, optional): Function tolerance for SLSQP.

    Returns:
        SyntheticDifferenceInDifferencesResult: Estimated SDID result container.
    """
    sdid = SyntheticDifferenceInDifferences(
        data=data,
        outcome_column=outcome_column,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
    )
    return sdid.fit(
        alpha=alpha,
        time_weight_ridge_scale=time_weight_ridge_scale,
        covariate_columns=covariate_columns,
        use_fixed_effects=use_fixed_effects,
        cluster_by_unit=cluster_by_unit,
        cluster_column=cluster_column,
        optimizer_max_iter=optimizer_max_iter,
        optimizer_tolerance=optimizer_tolerance,
    )


def fit_unit_weights(
    panel: pd.DataFrame,
    outcome_col: str,
    unit_column: str = "unit",
    time_column: str = "time",
    treatment_column: str = "treated",
    post_column: str = "post",
    optimizer_max_iter: int = 5000,
    optimizer_tolerance: float = 1e-9,
) -> pd.Series:
    """Low-level helper to estimate SDID unit weights.

    Args:
        panel (pd.DataFrame): Input panel dataframe.
        outcome_col (str): Outcome column.
        unit_column (str, optional): Unit identifier column.
        time_column (str, optional): Time column.
        treatment_column (str, optional): Treatment indicator column.
        post_column (str, optional): Post-period indicator column.
        optimizer_max_iter (int, optional): Maximum SLSQP iterations.
        optimizer_tolerance (float, optional): Function tolerance for SLSQP.

    Returns:
        pd.Series: Estimated unit weights indexed by control units.
    """
    prepared_panel = _prepare_sdid_panel(
        data=panel,
        outcome_column=outcome_col,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
    )
    unit_weights, _ = _fit_unit_weights_core(
        panel=prepared_panel,
        outcome_column=outcome_col,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
        optimizer_max_iter=optimizer_max_iter,
        optimizer_tolerance=optimizer_tolerance,
    )
    return unit_weights.sort_values(ascending=False)


def fit_time_weights(
    panel: pd.DataFrame,
    outcome_col: str,
    unit_column: str = "unit",
    time_column: str = "time",
    treatment_column: str = "treated",
    post_column: str = "post",
    zeta_time_small: float | None = None,
    optimizer_max_iter: int = 5000,
    optimizer_tolerance: float = 1e-9,
) -> pd.Series:
    """Low-level helper to estimate SDID time weights.

    Args:
        panel (pd.DataFrame): Input panel dataframe.
        outcome_col (str): Outcome column.
        unit_column (str, optional): Unit identifier column.
        time_column (str, optional): Time column.
        treatment_column (str, optional): Treatment indicator column.
        post_column (str, optional): Post-period indicator column.
        zeta_time_small (float | None, optional): Explicit time regularization value.
        optimizer_max_iter (int, optional): Maximum SLSQP iterations.
        optimizer_tolerance (float, optional): Function tolerance for SLSQP.

    Returns:
        pd.Series: Estimated time weights indexed by pre-period timestamps.
    """
    prepared_panel = _prepare_sdid_panel(
        data=panel,
        outcome_column=outcome_col,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
    )
    ridge_scale = 1e-6
    if zeta_time_small is not None:
        sigma_hat = calculate_sigma_hat(
            panel=prepared_panel,
            outcome_col=outcome_col,
            unit_column=unit_column,
            time_column=time_column,
            treatment_column=treatment_column,
            post_column=post_column,
        )
        ridge_scale = (
            0.0 if np.isclose(sigma_hat, 0.0) else float(zeta_time_small / sigma_hat)
        )
    time_weights, _ = _fit_time_weights_core(
        panel=prepared_panel,
        outcome_column=outcome_col,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
        time_weight_ridge_scale=ridge_scale,
        optimizer_max_iter=optimizer_max_iter,
        optimizer_tolerance=optimizer_tolerance,
    )
    return time_weights.sort_index()


def join_sdid_weights(
    panel: pd.DataFrame,
    unit_weights: pd.Series,
    time_weights: pd.Series,
    unit_column: str = "unit",
    time_column: str = "time",
    treatment_column: str = "treated",
    post_column: str = "post",
    outcome_column: str = "outcome",
) -> pd.DataFrame:
    """Low-level helper to merge SDID unit/time weights into panel.

    Args:
        panel (pd.DataFrame): Input panel dataframe.
        unit_weights (pd.Series): Unit weights indexed by unit.
        time_weights (pd.Series): Time weights indexed by time.
        unit_column (str, optional): Unit identifier column.
        time_column (str, optional): Time column.
        treatment_column (str, optional): Treatment indicator column.
        post_column (str, optional): Post-period indicator column.
        outcome_column (str, optional): Outcome column name used for validation.

    Returns:
        pd.DataFrame: Panel with merged SDID weights.
    """
    if outcome_column not in panel.columns:
        # outcome value is not needed for merge itself, but class constructor validates it.
        panel = panel.copy()
        panel[outcome_column] = 0.0
    prepared_panel = _prepare_sdid_panel(
        data=panel,
        outcome_column=outcome_column,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
    )
    return _join_sdid_weights_core(
        panel=prepared_panel,
        unit_weights=unit_weights,
        time_weights=time_weights,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
    )


def fit_sdid(
    panel: pd.DataFrame,
    outcome_col: str,
    unit_column: str = "unit",
    time_column: str = "time",
    treatment_column: str = "treated",
    post_column: str = "post",
    covariate_cols: list[str] | None = None,
    alpha: float = 0.05,
    time_weight_ridge_scale: float = 1e-6,
    use_fixed_effects: bool = True,
    cluster_by_unit: bool = True,
    cluster_column: str | None = None,
    optimizer_max_iter: int = 5000,
    optimizer_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compatibility wrapper similar to sdid.py fit_sdid(...) return format.

    Args:
        panel (pd.DataFrame): Input panel dataframe.
        outcome_col (str): Outcome column.
        unit_column (str, optional): Unit identifier column.
        time_column (str, optional): Time column.
        treatment_column (str, optional): Treatment indicator column.
        post_column (str, optional): Post-period indicator column.
        covariate_cols (list[str] | None, optional): Optional residualization covariates.
        alpha (float, optional): Significance level.
        time_weight_ridge_scale (float, optional): Time-weight regularization scale.
        use_fixed_effects (bool, optional): Use unit/time fixed effects in final regression.
        cluster_by_unit (bool, optional): Use clustered standard errors.
        cluster_column (str | None, optional): Cluster column name.
        optimizer_max_iter (int, optional): Maximum SLSQP iterations.
        optimizer_tolerance (float, optional): Function tolerance for SLSQP.

    Returns:
        dict[str, Any]: Compatibility payload with ATT, p-value, weights and metadata.
    """
    result = synthetic_difference_in_differences(
        data=panel,
        outcome_column=outcome_col,
        unit_column=unit_column,
        time_column=time_column,
        treatment_column=treatment_column,
        post_column=post_column,
        alpha=alpha,
        time_weight_ridge_scale=time_weight_ridge_scale,
        covariate_columns=covariate_cols,
        use_fixed_effects=use_fixed_effects,
        cluster_by_unit=cluster_by_unit,
        cluster_column=cluster_column,
        optimizer_max_iter=optimizer_max_iter,
        optimizer_tolerance=optimizer_tolerance,
    )
    return {
        "att": result.estimate,
        "model": result,
        "unit_weights": result.unit_weights.sort_values(ascending=False),
        "time_weights": result.time_weights.sort_index(),
        "did_data": result.weighted_data.copy(),
        "pvalue": float(result.pvalue),
        "significance": bool(
            np.isfinite(result.pvalue) and float(result.pvalue) < float(alpha)
        ),
        "outcome_col": outcome_col,
        "model_outcome_col": result.model_outcome_column,
        "residualized": result.residualized,
        "covariate_cols": result.covariate_columns,
        "residual_model": result.residualization_model,
        "time_weight_ridge": result.time_regularization,
        "used_fixed_effects": result.used_fixed_effects,
        "used_clustered_se": result.used_clustered_se,
        "cluster_column": result.cluster_column,
    }


def fit_staggered_sdid(
    data: pd.DataFrame,
    outcome_col: str,
    unit_column: str = "unit",
    time_column: str = "time",
    treatment_column: str = "treated",
    post_column: str = "post",
    treatment_start_column: str | None = None,
    include_not_yet_treated_as_controls: bool = True,
    alpha: float = 0.05,
    covariate_cols: list[str] | None = None,
    time_weight_ridge_scale: float = 1e-6,
    use_fixed_effects: bool = True,
    cluster_by_unit: bool = True,
    cluster_column: str | None = None,
    optimizer_max_iter: int = 5000,
    optimizer_tolerance: float = 1e-9,
    continue_on_error: bool = True,
    weighting_column: str = "n_treated_post_obs",
) -> StaggeredSyntheticDifferenceInDifferencesResult:
    """Generic staggered-adoption SDID via cohort decomposition.

    Args:
        data (pd.DataFrame): Input panel dataframe.
        outcome_col (str): Outcome column.
        unit_column (str, optional): Unit identifier column.
        time_column (str, optional): Time column.
        treatment_column (str, optional): Treatment indicator column.
        post_column (str, optional): Post-period indicator column.
        treatment_start_column (str | None, optional): Optional explicit treatment-start column.
        include_not_yet_treated_as_controls (bool, optional): Include not-yet-treated units
            as controls for earlier cohorts.
        alpha (float, optional): Significance level.
        covariate_cols (list[str] | None, optional): Optional residualization covariates.
        time_weight_ridge_scale (float, optional): Time-weight regularization scale.
        use_fixed_effects (bool, optional): Use unit/time fixed effects in final regression.
        cluster_by_unit (bool, optional): Use clustered standard errors.
        cluster_column (str | None, optional): Cluster column name.
        optimizer_max_iter (int, optional): Maximum SLSQP iterations.
        optimizer_tolerance (float, optional): Function tolerance for SLSQP.
        continue_on_error (bool, optional): Skip broken cohorts instead of raising.
        weighting_column (str, optional): Column used for aggregate weighted average.

    Returns:
        StaggeredSyntheticDifferenceInDifferencesResult: Cohort-level and aggregated SDID outputs.
    """
    required_columns = [
        unit_column,
        time_column,
        outcome_col,
        treatment_column,
        post_column,
    ]
    if treatment_start_column is not None:
        required_columns.append(treatment_start_column)
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    panel = data.copy()
    panel[unit_column] = panel[unit_column].astype(str)
    panel = panel.dropna(
        subset=[unit_column, time_column, outcome_col, treatment_column, post_column]
    ).copy()
    if panel.empty:
        raise ValueError("No rows left after dropping NaN in required columns.")

    if treatment_start_column is not None:
        starts = (
            panel[[unit_column, treatment_start_column]]
            .dropna(subset=[treatment_start_column])
            .groupby(unit_column)[treatment_start_column]
            .min()
        )
    else:
        starts = (
            panel[
                (panel[treatment_column].astype(int) == 1)
                & (panel[post_column].astype(int) == 1)
            ]
            .groupby(unit_column)[time_column]
            .min()
        )

    all_units = sorted(panel[unit_column].drop_duplicates().tolist())
    treatment_start_by_unit = pd.Series(index=all_units, dtype=object)
    treatment_start_by_unit.loc[starts.index.astype(str)] = starts.values

    cohort_dates = sorted(
        [x for x in treatment_start_by_unit.dropna().unique().tolist()]
    )
    if len(cohort_dates) == 0:
        raise ValueError("No treated cohorts found for staggered SDID.")

    rows: list[dict[str, Any]] = []
    cohort_models: dict[Any, SyntheticDifferenceInDifferencesResult] = {}

    for cohort_date in cohort_dates:
        cohort_units = (
            treatment_start_by_unit[treatment_start_by_unit == cohort_date]
            .index.astype(str)
            .tolist()
        )
        if not cohort_units:
            continue

        never_treated_units = treatment_start_by_unit[
            treatment_start_by_unit.isna()
        ].index.tolist()
        later_treated_units = treatment_start_by_unit[
            treatment_start_by_unit.notna() & (treatment_start_by_unit > cohort_date)
        ].index.tolist()
        control_units = list(never_treated_units)
        if include_not_yet_treated_as_controls:
            control_units += list(later_treated_units)
        control_units = sorted(list(set(control_units)))
        if len(control_units) == 0:
            if continue_on_error:
                continue
            raise ValueError(f"Cohort {cohort_date}: no control units available.")

        selected_units = sorted(list(set(cohort_units + control_units)))
        cohort_panel = panel[panel[unit_column].isin(selected_units)].copy()
        cohort_panel["_cohort_start"] = cohort_panel[unit_column].map(
            treatment_start_by_unit
        )
        if include_not_yet_treated_as_controls:
            # Keep control rows only while they are not-yet-treated at this cohort date.
            mask_drop = (
                ~cohort_panel[unit_column].isin(cohort_units)
                & cohort_panel["_cohort_start"].notna()
                & (cohort_panel[time_column] >= cohort_panel["_cohort_start"])
            )
            cohort_panel = cohort_panel[~mask_drop].copy()
        cohort_panel["cohort_treated"] = (
            cohort_panel[unit_column].isin(cohort_units).astype(int)
        )
        cohort_panel["cohort_post"] = (cohort_panel[time_column] >= cohort_date).astype(
            int
        )
        cohort_panel = cohort_panel.drop(columns=["_cohort_start"])

        try:
            # Reuse one-shot API to avoid duplicating class-construction + fit logic.
            cohort_result = synthetic_difference_in_differences(
                data=cohort_panel,
                outcome_column=outcome_col,
                unit_column=unit_column,
                time_column=time_column,
                treatment_column="cohort_treated",
                post_column="cohort_post",
                alpha=alpha,
                time_weight_ridge_scale=time_weight_ridge_scale,
                covariate_columns=covariate_cols,
                use_fixed_effects=use_fixed_effects,
                cluster_by_unit=cluster_by_unit,
                cluster_column=cluster_column
                if cluster_column is not None
                else unit_column,
                optimizer_max_iter=optimizer_max_iter,
                optimizer_tolerance=optimizer_tolerance,
            )
        except Exception:
            if continue_on_error:
                continue
            raise

        n_treated_post_obs = int(
            (
                (cohort_panel["cohort_treated"] == 1)
                & (cohort_panel["cohort_post"] == 1)
            ).sum()
        )
        row = {
            "cohort_date": cohort_date,
            "estimate": cohort_result.estimate,
            "std_error": cohort_result.std_error,
            "pvalue": cohort_result.pvalue,
            "ci_lower": cohort_result.ci_lower,
            "ci_upper": cohort_result.ci_upper,
            "n_obs": cohort_result.n_obs,
            "n_treated_units": len(cohort_units),
            "n_control_units": len(control_units),
            "n_treated_post_obs": n_treated_post_obs,
        }
        rows.append(row)
        cohort_models[cohort_date] = cohort_result

    if len(rows) == 0:
        raise ValueError("No cohort-specific SDID models were fitted.")
    cohort_results = (
        pd.DataFrame(rows).sort_values("cohort_date").reset_index(drop=True)
    )
    if weighting_column not in cohort_results.columns:
        raise ValueError(
            f"weighting_column '{weighting_column}' is not available in cohort_results."
        )
    weights = cohort_results[weighting_column].astype(float).to_numpy()
    if np.isclose(float(weights.sum()), 0.0):
        weights = np.ones_like(weights, dtype=float)

    aggregate_estimate = float(
        np.average(cohort_results["estimate"].astype(float), weights=weights)
    )
    return StaggeredSyntheticDifferenceInDifferencesResult(
        aggregate_estimate=aggregate_estimate,
        cohort_results=cohort_results,
        cohort_models=cohort_models,
        weighting_column=weighting_column,
    )
