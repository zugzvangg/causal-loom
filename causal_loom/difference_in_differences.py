from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from scipy import stats

from causal_loom._did_utils import safe_mean, validate_binary_column
from causal_loom.visualization import (
    create_did_2x2_figure,
    create_did_parallel_trends_figure,
)


@dataclass(frozen=True)
class DifferenceInDifferencesResult:
    """Container with classic 2x2 DiD estimate and inference outputs.

    Attributes:
        estimate (float): DiD estimate (interaction coefficient).
        std_error (float): Standard error of DiD estimate from OLS.
        t_statistic (float): t-statistic for DiD estimate.
        pvalue (float): Two-sided p-value for DiD estimate.
        ci_lower (float): Lower confidence bound.
        ci_upper (float): Upper confidence bound.
        alpha (float): Significance level used for confidence interval.
        n_obs (int): Number of observations used in estimation.
        group_means (pd.DataFrame): Table with mean outcome by treatment/post cells.
        counterfactual_treated_post (float): Predicted treated-post mean under no treatment.
        treated_change (float): Before/after change in treated group.
        control_change (float): Before/after change in control group.
        naive_before_after (float): Naive before/after estimate (treated only).
        naive_post_period_comparison (float): Naive post-period treated-vs-control estimate.
    """

    estimate: float
    std_error: float
    t_statistic: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    alpha: float
    n_obs: int
    group_means: pd.DataFrame
    counterfactual_treated_post: float
    treated_change: float
    control_change: float
    naive_before_after: float
    naive_post_period_comparison: float


class DifferenceInDifferences:
    """Difference-in-Differences estimator for 2x2 treatment-time setup.

    Args:
        data (pd.DataFrame): Input dataframe.
        outcome_column (str): Numeric outcome column.
        treatment_column (str): Binary treatment-group indicator (1 treated, 0 control).
        post_column (str): Binary post-period indicator (1 post, 0 pre).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome_column: str,
        treatment_column: str,
        post_column: str,
    ) -> None:
        self.data = data.copy()
        self.outcome_column = outcome_column
        self.treatment_column = treatment_column
        self.post_column = post_column
        self.latest_result: DifferenceInDifferencesResult | None = None

        self.__validate_inputs()
        self.__normalize_inputs()

    def __validate_inputs(self) -> None:
        required = [self.outcome_column, self.treatment_column, self.post_column]
        missing = [column for column in required if column not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def __normalize_inputs(self) -> None:
        prepared = self.data.copy()
        prepared = prepared.dropna(
            subset=[self.outcome_column, self.treatment_column, self.post_column]
        )
        if prepared.empty:
            raise ValueError("No rows left after dropping NaN in required columns.")

        if not pd.api.types.is_numeric_dtype(prepared[self.outcome_column]):
            raise ValueError(f"Column '{self.outcome_column}' should be numeric.")

        prepared[self.treatment_column] = prepared[self.treatment_column].astype(int)
        prepared[self.post_column] = prepared[self.post_column].astype(int)

        validate_binary_column(prepared, self.treatment_column)
        validate_binary_column(prepared, self.post_column)
        self.data = prepared.reset_index(drop=True)

    def fit(self, alpha: float = 0.05) -> DifferenceInDifferencesResult:
        """Estimate DiD effect and OLS-based inference.

        Args:
            alpha (float, optional): Significance level for confidence interval.

        Returns:
            DifferenceInDifferencesResult: Estimated effect and diagnostics.
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha should be in (0, 1); got {alpha}.")

        d = self.data[self.treatment_column].astype(float).to_numpy()
        t = self.data[self.post_column].astype(float).to_numpy()
        y = self.data[self.outcome_column].astype(float).to_numpy()

        treated_pre = safe_mean(
            self.data,
            (self.data[self.treatment_column] == 1)
            & (self.data[self.post_column] == 0),
            self.outcome_column,
        )
        treated_post = safe_mean(
            self.data,
            (self.data[self.treatment_column] == 1)
            & (self.data[self.post_column] == 1),
            self.outcome_column,
        )
        control_pre = safe_mean(
            self.data,
            (self.data[self.treatment_column] == 0)
            & (self.data[self.post_column] == 0),
            self.outcome_column,
        )
        control_post = safe_mean(
            self.data,
            (self.data[self.treatment_column] == 0)
            & (self.data[self.post_column] == 1),
            self.outcome_column,
        )

        interaction = d * t
        X = np.column_stack([np.ones_like(d), d, t, interaction])
        n_obs, n_params = X.shape
        if n_obs <= n_params:
            raise ValueError(
                f"Need more observations than parameters for OLS inference, got n={n_obs}."
            )

        xtx_inv = np.linalg.inv(X.T @ X)
        beta = xtx_inv @ (X.T @ y)
        residuals = y - X @ beta
        sigma2 = float((residuals @ residuals) / (n_obs - n_params))
        cov = sigma2 * xtx_inv
        std_error = float(np.sqrt(cov[3, 3]))
        estimate = float(beta[3])
        if np.isclose(std_error, 0.0):
            t_stat = np.nan
            pvalue = np.nan
            ci_delta = np.nan
        else:
            t_stat = float(estimate / std_error)
            pvalue = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n_obs - n_params)))
            ci_delta = float(
                stats.t.ppf(1 - alpha / 2, df=n_obs - n_params) * std_error
            )

        group_means = pd.DataFrame(
            {
                "group": ["control", "control", "treated", "treated"],
                "post_period": [0, 1, 0, 1],
                "mean_outcome": [control_pre, control_post, treated_pre, treated_post],
            }
        )
        treated_change = float(treated_post - treated_pre)
        control_change = float(control_post - control_pre)
        naive_before_after = treated_change
        naive_post_period_comparison = float(treated_post - control_post)
        counterfactual_treated_post = float(treated_pre + (control_post - control_pre))

        result = DifferenceInDifferencesResult(
            estimate=estimate,
            std_error=std_error,
            t_statistic=t_stat,
            pvalue=pvalue,
            ci_lower=float(estimate - ci_delta),
            ci_upper=float(estimate + ci_delta),
            alpha=alpha,
            n_obs=int(n_obs),
            group_means=group_means,
            counterfactual_treated_post=counterfactual_treated_post,
            treated_change=treated_change,
            control_change=control_change,
            naive_before_after=naive_before_after,
            naive_post_period_comparison=naive_post_period_comparison,
        )
        self.latest_result = result
        return result

    def get_estimators_breakdown(self) -> pd.DataFrame:
        """Return naive estimators and DiD decomposition in one table.

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
                    "did_controlled_difference",
                ],
                "value": [
                    result.naive_before_after,
                    result.naive_post_period_comparison,
                    result.estimate,
                ],
            }
        )

    def get_parallel_trends_frame(
        self,
        time_column: str,
        pre_period_only: bool = True,
    ) -> pd.DataFrame:
        """Prepare aggregated treated/control means by time for trend diagnostics.

        Args:
            time_column (str): Time column in original dataframe.
            pre_period_only (bool, optional): Keep only pre-period rows if True.

        Returns:
            pd.DataFrame: Dataframe with columns time, treated_mean, control_mean, gap.
        """
        if time_column not in self.data.columns:
            raise ValueError(f"Column '{time_column}' not found in data.")

        source = self.data.copy()
        source[time_column] = pd.to_datetime(source[time_column], errors="coerce")
        source = source.dropna(subset=[time_column])
        if source.empty:
            raise ValueError(f"Column '{time_column}' has no valid datetime values.")
        if pre_period_only:
            source = source[source[self.post_column] == 0].copy()
            if source.empty:
                raise ValueError("No pre-period rows available for trend diagnostics.")

        grouped = (
            source.groupby([time_column, self.treatment_column], as_index=False)[
                self.outcome_column
            ]
            .mean()
            .pivot(
                index=time_column,
                columns=self.treatment_column,
                values=self.outcome_column,
            )
            .rename(columns={0: "control_mean", 1: "treated_mean"})
            .reset_index()
            .sort_values(time_column)
        )
        if (
            "control_mean" not in grouped.columns
            or "treated_mean" not in grouped.columns
        ):
            raise ValueError(
                "Trend diagnostics need both treatment groups in selected period."
            )
        grouped["gap"] = grouped["treated_mean"] - grouped["control_mean"]
        return grouped.rename(columns={time_column: "time"})

    def plot_2x2(self) -> go.Figure:
        """Plot classic 2x2 DiD chart with counterfactual treated-post point.

        Returns:
            go.Figure: Plotly figure.
        """
        if self.latest_result is None:
            raise ValueError("Run fit() before plotting 2x2 DiD chart.")
        fig = create_did_2x2_figure(
            group_means=self.latest_result.group_means,
            counterfactual_treated_post=self.latest_result.counterfactual_treated_post,
            estimate=self.latest_result.estimate,
        )
        fig.show()
        return fig

    def plot_parallel_trends(
        self,
        time_column: str,
        treatment_start: Union[str, pd.Timestamp, None] = None,
        pre_period_only: bool = False,
    ) -> go.Figure:
        """Plot treated/control trends by time for visual parallel-trend check.

        Args:
            time_column (str): Time column in source dataframe.
            treatment_start (Union[str, pd.Timestamp, None], optional): Vertical marker date.
            pre_period_only (bool, optional): Plot only pre-period rows if True.

        Returns:
            go.Figure: Plotly figure.
        """
        source = self.data.copy()
        if time_column not in source.columns:
            raise ValueError(
                f"Column '{time_column}' not found in data used by DifferenceInDifferences."
            )
        source[time_column] = pd.to_datetime(source[time_column])
        if pre_period_only:
            source = source[source[self.post_column] == 0].copy()

        grouped = (
            source.groupby([time_column, self.treatment_column], as_index=False)[
                self.outcome_column
            ]
            .mean()
            .pivot(
                index=time_column,
                columns=self.treatment_column,
                values=self.outcome_column,
            )
            .rename(columns={0: "control_mean", 1: "treated_mean"})
            .reset_index()
            .sort_values(time_column)
        )
        if (
            "control_mean" not in grouped.columns
            or "treated_mean" not in grouped.columns
        ):
            raise ValueError("Need both treatment groups to plot parallel trends.")

        fig = create_did_parallel_trends_figure(
            trends_frame=grouped.rename(columns={time_column: "time"}),
            treatment_start=treatment_start,
            outcome_column=self.outcome_column,
            pre_period_only=pre_period_only,
        )
        fig.show()
        return fig


def difference_in_differences(
    data: pd.DataFrame,
    outcome_column: str,
    treatment_column: str,
    post_column: str,
    alpha: float = 0.05,
) -> DifferenceInDifferencesResult:
    """Convenience function for one-shot 2x2 DiD estimation.

    Args:
        data (pd.DataFrame): Input dataframe.
        outcome_column (str): Numeric outcome column.
        treatment_column (str): Binary treatment-group indicator.
        post_column (str): Binary post-period indicator.
        alpha (float, optional): Significance level for confidence interval.

    Returns:
        DifferenceInDifferencesResult: Estimated effect and inference outputs.
    """
    did = DifferenceInDifferences(
        data=data,
        outcome_column=outcome_column,
        treatment_column=treatment_column,
        post_column=post_column,
    )
    return did.fit(alpha=alpha)
