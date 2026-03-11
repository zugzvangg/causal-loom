import datetime
from dataclasses import dataclass
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from tqdm import tqdm

from cib_advanced_experiments.synthetic_control import SyntheticControl
from cib_advanced_experiments.visualization import (
    create_experiment_aggregated_series_figure,
    create_experiment_distributions_figure,
)


@dataclass(frozen=True)
class FittedSyntheticControlResult:
    """Container for one fitted synthetic-control model.

    Attributes:
        treated_unit (str): Treated unit identifier.
        treatment_date (pd.Timestamp): Treatment date.
        synthetic_control (SyntheticControl): Fitted synthetic-control object.
        fit_error (float): Model fit error value.
    """

    treated_unit: str
    treatment_date: pd.Timestamp
    synthetic_control: SyntheticControl
    fit_error: float


@dataclass(frozen=True)
class ExperimentResult:
    """Container for experiment-level outputs and inference.

    Attributes:
        fitted_units (List[FittedSyntheticControlResult]): Per-unit fitted models.
        unit_level_frame (pd.DataFrame): Per-unit metrics before fit-quality filtering.
        filtered_frame (pd.DataFrame): Per-unit metrics after fit-quality filtering.
        border_quantile (float): Fit-error quantile threshold.
        fit_errors (List[float]): Fit errors for all fitted treated units.
        included_units (List[str]): Units used in final inference.
        excluded_units (List[str]): Units excluded by fit-quality filter.
        mean_synthetic (float): Mean synthetic value after filtering.
        mean_real (float): Mean real value after filtering.
        expected_effect_in_percent (float): Relative effect in percent.
        test_name (str): Name of inference test.
        test_statistic (float): Test statistic value.
        test_pvalue (float): P-value.
        theta (float | None): CUPED theta if applied.
        effect_evaluation_aggregation (str): Aggregation mode used for effect.
    """

    fitted_units: List[FittedSyntheticControlResult]
    unit_level_frame: pd.DataFrame
    filtered_frame: pd.DataFrame
    border_quantile: float
    fit_errors: List[float]
    included_units: List[str]
    excluded_units: List[str]
    mean_synthetic: float
    mean_real: float
    expected_effect_in_percent: float
    test_name: str
    test_statistic: float
    test_pvalue: float
    theta: float | None
    effect_evaluation_aggregation: str


class Experiment:
    """Run multiple synthetic-control fits and aggregate experiment-level inference.

    Args:
        data (pd.DataFrame): Source panel data.
        treatments_dataframe (pd.DataFrame): Table with treated units and treatment dates.
        unit_column (str): Unit identifier column name.
        date_column (str): Date column name.
        features (List[str]): Features/outcomes available for modeling.
        treatment_date_column (str): Treatment date column in treatments_dataframe.
        min_date (Union[str, datetime.datetime], optional): Lower date bound.
        max_date (Union[str, datetime.datetime], optional): Upper date bound.
        date_frequency (Literal["D", "W", "M", "Y"], optional): Aggregation frequency.
        aggregation_statistic (Literal["sum", "mean", "max", "min"], optional): Aggregation operation.
        cuped_enabled (bool, optional): Whether to apply CUPED adjustment.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatments_dataframe: pd.DataFrame,
        unit_column: str,
        date_column: str,
        features: List[str],
        treatment_date_column: str,
        min_date: Union[str, datetime.datetime] = None,
        max_date: Union[str, datetime.datetime] = None,
        date_frequency: Literal["D", "W", "M", "Y"] = "D",
        aggregation_statistic: Literal["sum", "mean", "max", "min"] = "mean",
        cuped_enabled: bool = False,
    ):
        """Initialize experiment object and validate treatment table.

        Args:
            data (pd.DataFrame): Source panel data.
            treatments_dataframe (pd.DataFrame): Table with treated units and treatment dates.
            unit_column (str): Unit identifier column name.
            date_column (str): Date column name.
            features (List[str]): Features/outcomes available for modeling.
            treatment_date_column (str): Treatment date column in treatments_dataframe.
            min_date (Union[str, datetime.datetime], optional): Lower date bound.
            max_date (Union[str, datetime.datetime], optional): Upper date bound.
            date_frequency (Literal["D", "W", "M", "Y"], optional): Aggregation frequency.
            aggregation_statistic (Literal["sum", "mean", "max", "min"], optional): Aggregation operation.
            cuped_enabled (bool, optional): Whether to apply CUPED adjustment.

        Returns:
            None
        """
        self.data = data
        self.treatments_dataframe = treatments_dataframe.copy()
        self.treatment_date_column = treatment_date_column
        self.unit_column = unit_column

        self.__validate_treatments_dataframe()

        self.date_column = date_column
        self.features = features
        self.min_date = min_date
        self.max_date = max_date
        self.date_frequency = date_frequency
        self.aggregation_statistic = aggregation_statistic
        self.resulting_synthetic_controls = None
        self.cuped_enabled = cuped_enabled
        self.latest_result: ExperimentResult | None = None

    def __validate_treatments_dataframe(self):
        """Validate treatments dataframe schema and uniqueness.

        Returns:
            None
        """
        treatments_dataframe_axis = self.treatments_dataframe.columns.values
        if self.treatment_date_column not in treatments_dataframe_axis:
            raise ValueError(
                f"No {self.treatment_date_column} in treatments dataframe axis={treatments_dataframe_axis}"
            )
        if self.unit_column not in treatments_dataframe_axis:
            raise ValueError(
                f"No {self.unit_column} in treatments dataframe axis={treatments_dataframe_axis}"
            )
        duplicated_units = self.treatments_dataframe[self.unit_column].duplicated()
        if duplicated_units.any():
            duplicates = (
                self.treatments_dataframe.loc[duplicated_units, self.unit_column]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                f"Each treated unit should appear once in treatments_dataframe. Duplicates: {duplicates}"
            )

    def fit_all(
        self,
        feature: str,
        top_weights_to_plot: int = 5,
        number_of_units_for_matching: int = None,
        categorical_matching: List[str] = None,
        numeric_matching: List[str] = None,
        matching_method: Literal["feature", "propensity"] = "feature",
        propensity_covariates: List[str] = None,
        propensity_caliper: float = None,
        propensity_random_state: int = 42,
        to_plot: bool = True,
        to_plot_aggregated: bool = False,
        raise_on_empty: bool = True,
        **kwargs,
    ) -> List[FittedSyntheticControlResult]:
        """Fit synthetic-control model for each treated unit.

        Args:
            feature (str): Target feature to evaluate.
            top_weights_to_plot (int, optional): Number of top weights for per-unit plots.
            number_of_units_for_matching (int, optional): Number of matched controls.
            categorical_matching (List[str], optional): Exact matching columns.
            numeric_matching (List[str], optional): Numeric matching columns.
            matching_method (Literal["feature", "propensity"], optional): Matching strategy.
            propensity_covariates (List[str], optional): Covariates for propensity matching.
            propensity_caliper (float, optional): Propensity caliper.
            propensity_random_state (int, optional): Propensity model random state.
            to_plot (bool, optional): Whether to plot per-unit synthetic-control results.
            to_plot_aggregated (bool, optional): Whether to plot aggregated mean real-vs-synthetic series after fitting all units.
            raise_on_empty (bool, optional): Raise error when no models were fitted.
            **kwargs: Additional parameters passed to SyntheticControl.

        Returns:
            List[FittedSyntheticControlResult]: Fitted model containers for treated units.
        """
        if feature not in self.features:
            raise ValueError(f"{feature=} should be from {self.features=}")

        fitted_results: List[FittedSyntheticControlResult] = []
        for treated_unit in tqdm(
            self.treatments_dataframe[self.unit_column].astype(str).unique(),
            desc="Synth for treated_unit",
        ):
            if treated_unit not in self.data[self.unit_column].astype(str).unique():
                logger.info(f"{treated_unit} not in history DataFrame. Skip fitting it.")
                continue

            treatment_date = pd.to_datetime(
                self.treatments_dataframe[
                    self.treatments_dataframe[self.unit_column].astype(str)
                    == treated_unit
                ]
                .iloc[0][self.treatment_date_column]
            )

            synthetic_control = SyntheticControl(
                data=self.data,
                unit_column=self.unit_column,
                date_column=self.date_column,
                features=self.features,
                treated_unit=treated_unit,
                treatment_date=treatment_date,
                min_date=self.min_date,
                max_date=self.max_date,
                number_of_units_for_matching=number_of_units_for_matching,
                categorical_matching=categorical_matching,
                numeric_matching=numeric_matching,
                matching_method=matching_method,
                propensity_covariates=propensity_covariates,
                propensity_caliper=propensity_caliper,
                propensity_random_state=propensity_random_state,
                date_frequency=self.date_frequency,
                aggregation_statistic=self.aggregation_statistic,
                **kwargs,
            )
            synthetic_control.fit()
            if to_plot:
                synthetic_control.plot_results(feature, top_weights_to_plot)

            fit_error = float(synthetic_control.model.get_loss_value())
            fitted_results.append(
                FittedSyntheticControlResult(
                    treated_unit=treated_unit,
                    treatment_date=treatment_date,
                    synthetic_control=synthetic_control,
                    fit_error=fit_error,
                )
            )

        self.resulting_synthetic_controls = [
            fitted.synthetic_control for fitted in fitted_results
        ]
        if len(fitted_results) == 0 and raise_on_empty:
            raise ValueError(
                "No synthetic controls were fitted. Check treatments_dataframe, "
                "matching settings, and unit coverage in source data."
            )
        if to_plot_aggregated and len(fitted_results) > 0:
            self.plot_aggregated_results(feature=feature, fitted_units=fitted_results)
        return fitted_results

    def get_aggregated_series(
        self,
        feature: str,
        fitted_units: List[FittedSyntheticControlResult] = None,
    ) -> pd.DataFrame:
        """Build experiment-level mean time series across fitted treated units.

        Args:
            feature (str): Target feature to aggregate.
            fitted_units (List[FittedSyntheticControlResult], optional): Explicit fitted results.
                If None, uses the latest fitted controls stored in this Experiment.

        Returns:
            pd.DataFrame: Dataframe with date, real, synthetic and effect columns.
        """
        if feature not in self.features:
            raise ValueError(f"{feature=} should be from {self.features=}")

        if fitted_units is None:
            if not self.resulting_synthetic_controls:
                raise ValueError(
                    "No fitted synthetic controls found. Run fit_all first or pass fitted_units."
                )
            synthetic_controls = self.resulting_synthetic_controls
        else:
            synthetic_controls = [f.synthetic_control for f in fitted_units]

        if len(synthetic_controls) == 0:
            raise ValueError("No fitted synthetic controls to aggregate.")

        per_unit_frames = []
        for synthetic_control in synthetic_controls:
            unit_frame = synthetic_control.get_data_on_features(feature=feature).copy()
            unit_frame["date"] = pd.to_datetime(unit_frame["date"])
            per_unit_frames.append(unit_frame[["date", "real", "synthetic"]])

        aggregated = (
            pd.concat(per_unit_frames, ignore_index=True)
            .groupby("date", as_index=False)[["real", "synthetic"]]
            .mean()
            .sort_values("date")
        )
        aggregated["effect"] = aggregated["real"] - aggregated["synthetic"]
        return aggregated

    def plot_aggregated_results(
        self,
        feature: str,
        fitted_units: List[FittedSyntheticControlResult] = None,
    ) -> None:
        """Plot mean real-vs-synthetic trajectory across fitted treated units.

        Args:
            feature (str): Target feature to plot.
            fitted_units (List[FittedSyntheticControlResult], optional): Explicit fitted results.
                If None, uses the latest fitted controls stored in this Experiment.

        Returns:
            None
        """
        aggregated = self.get_aggregated_series(
            feature=feature,
            fitted_units=fitted_units,
        )

        if fitted_units is None:
            synthetic_controls = self.resulting_synthetic_controls
        else:
            synthetic_controls = [f.synthetic_control for f in fitted_units]
        treatment_dates = sorted(
            {pd.to_datetime(sc.treatment_date) for sc in synthetic_controls}
        )
        treatment_date_for_marker = None
        if len(treatment_dates) == 1:
            treatment_date_for_marker = treatment_dates[0]
        elif len(treatment_dates) > 1:
            logger.warning(
                "Multiple treatment dates detected across fitted units. "
                "Aggregated plot is shown without a single treatment-date marker."
            )

        fig = create_experiment_aggregated_series_figure(
            df_to_plot=aggregated,
            feature=feature,
            treated_units_count=len(synthetic_controls),
            treatment_date=treatment_date_for_marker,
        )
        fig.show()

    def evaluate(
        self,
        fitted_units: List[FittedSyntheticControlResult],
        feature: str,
        effect_evaluation_aggregation: Literal["last_date", "mean_value"] = "mean_value",
        quantile_error_border: float = 0.95,
        min_units_after_filter: int = 1,
        min_units_for_inference: int = 1,
        inference_method: Literal[
            "independent_ttest", "paired_ttest", "permutation"
        ] = "independent_ttest",
        permutation_iterations: int = 2000,
        random_state: int = 42,
        raise_on_empty: bool = True,
        to_plot: bool = True,
    ) -> ExperimentResult:
        """Aggregate fitted units into experiment-level metrics and inference.

        Args:
            fitted_units (List[FittedSyntheticControlResult]): Fitted unit-level models.
            feature (str): Target feature to evaluate.
            effect_evaluation_aggregation (Literal["last_date", "mean_value"], optional): Effect aggregation mode.
            quantile_error_border (float, optional): Fit-error filtering quantile in [0, 1].
            min_units_after_filter (int, optional): Minimum units to keep after quality filtering.
            min_units_for_inference (int, optional): Minimum units required for statistical inference.
            inference_method (Literal["independent_ttest", "paired_ttest", "permutation"], optional): Inference method.
            permutation_iterations (int, optional): Number of permutations for permutation test.
            random_state (int, optional): Random seed for permutation test.
            raise_on_empty (bool, optional): Raise if filtered units are empty.
            to_plot (bool, optional): Whether to show experiment-level distributions figure.

        Returns:
            ExperimentResult: Aggregated metrics and unit-level filtering artifacts.
        """
        if not (0.0 <= quantile_error_border <= 1.0):
            raise ValueError(
                f"quantile_error_border should be in [0,1], got {quantile_error_border}."
            )
        if min_units_after_filter <= 0:
            raise ValueError(
                f"min_units_after_filter should be positive, got {min_units_after_filter}."
            )
        if min_units_for_inference <= 0:
            raise ValueError(
                f"min_units_for_inference should be positive, got {min_units_for_inference}."
            )
        if not fitted_units:
            if raise_on_empty:
                raise ValueError("No fitted units were provided for evaluation.")
            fitted_units = []

        rows = []
        fit_errors = [f.fit_error for f in fitted_units]
        border_quantile = (
            float(np.quantile(fit_errors, quantile_error_border)) if fit_errors else np.nan
        )

        for fitted in fitted_units:
            data = fitted.synthetic_control.get_data_on_features(feature)
            if effect_evaluation_aggregation == "last_date":
                data = data[data["date"] == data["date"].max()]
            elif effect_evaluation_aggregation == "mean_value":
                data = data[data["date"] >= fitted.synthetic_control.treatment_date]
            else:
                raise ValueError(
                    "effect_evaluation_aggregation should be one of ['last_date', 'mean_value']"
                )

            rows.append(
                {
                    "treated_unit": fitted.treated_unit,
                    "fit_error": fitted.fit_error,
                    "real": float(data["real"].mean()),
                    "synthetic": float(data["synthetic"].mean()),
                    "pre_mean": float(
                        fitted.synthetic_control.get_pre_treatment_mean(feature)
                    ),
                }
            )

        unit_level_frame = pd.DataFrame(rows)
        if unit_level_frame.empty:
            if raise_on_empty:
                raise ValueError("No unit-level records were computed for evaluation.")
            filtered_frame = unit_level_frame.copy()
            excluded_units = []
        else:
            filtered_frame = unit_level_frame[
                unit_level_frame["fit_error"] <= border_quantile
            ].copy()
            if len(filtered_frame) < min_units_after_filter:
                logger.warning(
                    "Fit-quality filter left too few units; fallback keeps best units by fit_error."
                )
                filtered_frame = unit_level_frame.nsmallest(
                    min(min_units_after_filter, len(unit_level_frame)), "fit_error"
                ).copy()
            excluded_units = sorted(
                list(
                    set(unit_level_frame["treated_unit"].tolist())
                    - set(filtered_frame["treated_unit"].tolist())
                )
            )

        if filtered_frame.empty and raise_on_empty:
            raise ValueError(
                "No valid fitted samples left after fit-error filtering. "
                "Try relaxing quantile_error_border or improving control group quality."
            )

        included_units = filtered_frame["treated_unit"].tolist()
        if len(filtered_frame) < min_units_for_inference:
            raise ValueError(
                f"Need at least {min_units_for_inference} units for inference, got {len(filtered_frame)}."
            )

        real = filtered_frame["real"].tolist()
        synthetic = filtered_frame["synthetic"].tolist()
        pre_means = filtered_frame["pre_mean"].tolist()

        theta = None
        if self.cuped_enabled:
            synthetic, real, theta = self.__apply_cuped(
                synthetic=synthetic, real=real, pre_means=pre_means
            )

        mean_synthetic = float(np.mean(synthetic))
        mean_real = float(np.mean(real))
        if np.isclose(mean_synthetic, 0.0):
            expected_effect_in_percent = np.nan
            logger.warning(
                "mean_synthetic is close to zero; expected_effect_in_percent set to NaN."
            )
        else:
            expected_effect_in_percent = round(
                (mean_real - mean_synthetic) / mean_synthetic * 100, 3
            )

        test_name, test_stat, test_pvalue = self.__run_inference(
            synthetic=synthetic,
            real=real,
            inference_method=inference_method,
            permutation_iterations=permutation_iterations,
            random_state=random_state,
        )

        if to_plot:
            fig = create_experiment_distributions_figure(
                real=real,
                synthetic=synthetic,
                fit_errors=fit_errors,
                border_quantile=border_quantile,
            )
            fig.show()

        result = ExperimentResult(
            fitted_units=fitted_units,
            unit_level_frame=unit_level_frame,
            filtered_frame=filtered_frame,
            border_quantile=border_quantile,
            fit_errors=fit_errors,
            included_units=included_units,
            excluded_units=excluded_units,
            mean_synthetic=mean_synthetic,
            mean_real=mean_real,
            expected_effect_in_percent=expected_effect_in_percent,
            test_name=test_name,
            test_statistic=float(test_stat),
            test_pvalue=float(test_pvalue),
            theta=theta,
            effect_evaluation_aggregation=effect_evaluation_aggregation,
        )
        self.latest_result = result
        self.mean_synthetic = result.mean_synthetic
        self.mean_real = result.mean_real
        self.expected_effect_in_percent = result.expected_effect_in_percent
        self.ttest = type("TTestResultLike", (), {"statistic": result.test_statistic, "pvalue": result.test_pvalue})()
        self.theta = result.theta
        return result

    def run_experiment(
        self,
        feature: str,
        top_weights_to_plot: int = 5,
        effect_evaluation_aggregation: Literal[
            "last_date", "mean_value"
        ] = "mean_value",
        quantile_error_border: float = 0.95,
        number_of_units_for_matching: int = None,
        categorical_matching: List[str] = None,
        numeric_matching: List[str] = None,
        matching_method: Literal["feature", "propensity"] = "feature",
        propensity_covariates: List[str] = None,
        propensity_caliper: float = None,
        propensity_random_state: int = 42,
        min_units_after_filter: int = 1,
        min_units_for_inference: int = 1,
        inference_method: Literal[
            "independent_ttest", "paired_ttest", "permutation"
        ] = "independent_ttest",
        permutation_iterations: int = 2000,
        random_state: int = 42,
        raise_on_empty: bool = True,
        return_result: bool = True,
        to_plot: bool = True,
        to_plot_individual: bool = None,
        to_plot_aggregated: bool = False,
        **kwargs,
    ) -> ExperimentResult | None:
        """Run full experiment pipeline: fit models, evaluate, and optionally plot.

        Args:
            feature (str): Target feature to evaluate.
            top_weights_to_plot (int, optional): Top weights shown in per-unit plots.
            effect_evaluation_aggregation (Literal["last_date", "mean_value"], optional): Effect aggregation mode.
            quantile_error_border (float, optional): Fit-error filtering quantile.
            number_of_units_for_matching (int, optional): Number of controls for automatic matching.
            categorical_matching (List[str], optional): Exact matching columns.
            numeric_matching (List[str], optional): Numeric matching columns.
            matching_method (Literal["feature", "propensity"], optional): Matching strategy.
            propensity_covariates (List[str], optional): Covariates for propensity matching.
            propensity_caliper (float, optional): Caliper for propensity matching.
            propensity_random_state (int, optional): Random state for propensity matching.
            min_units_after_filter (int, optional): Minimum units retained after fit filter.
            min_units_for_inference (int, optional): Minimum units required for inference.
            inference_method (Literal["independent_ttest", "paired_ttest", "permutation"], optional): Inference method.
            permutation_iterations (int, optional): Number of permutations for permutation method.
            random_state (int, optional): Random seed for permutation method.
            raise_on_empty (bool, optional): Raise error for empty fit/filter states.
            return_result (bool, optional): Return ExperimentResult if True.
            to_plot (bool, optional): Plot final experiment-level distributions.
            to_plot_individual (bool, optional): Plot per-unit synthetic-control outputs. If None, equals to_plot.
            to_plot_aggregated (bool, optional): Plot aggregated mean real-vs-synthetic series after fitting all units.
            **kwargs: Additional kwargs passed to SyntheticControl.

        Returns:
            ExperimentResult | None: Experiment result if return_result=True, otherwise None.
        """
        if to_plot_individual is None:
            to_plot_individual = to_plot

        fitted_units = self.fit_all(
            feature=feature,
            top_weights_to_plot=top_weights_to_plot,
            number_of_units_for_matching=number_of_units_for_matching,
            categorical_matching=categorical_matching,
            numeric_matching=numeric_matching,
            matching_method=matching_method,
            propensity_covariates=propensity_covariates,
            propensity_caliper=propensity_caliper,
            propensity_random_state=propensity_random_state,
            to_plot=to_plot_individual,
            to_plot_aggregated=to_plot_aggregated,
            raise_on_empty=raise_on_empty,
            **kwargs,
        )
        result = self.evaluate(
            fitted_units=fitted_units,
            feature=feature,
            effect_evaluation_aggregation=effect_evaluation_aggregation,
            quantile_error_border=quantile_error_border,
            min_units_after_filter=min_units_after_filter,
            min_units_for_inference=min_units_for_inference,
            inference_method=inference_method,
            permutation_iterations=permutation_iterations,
            random_state=random_state,
            raise_on_empty=raise_on_empty,
            to_plot=to_plot,
        )
        return result if return_result else None

    @staticmethod
    def __apply_cuped(
        synthetic: List[float], real: List[float], pre_means: List[float]
    ) -> tuple[List[float], List[float], float | None]:
        """Apply CUPED adjustment when pre-period variance is informative.

        Args:
            synthetic (List[float]): Synthetic values.
            real (List[float]): Real values.
            pre_means (List[float]): Pre-period means per treated unit.

        Returns:
            tuple[List[float], List[float], float | None]: Adjusted synthetic, real and theta.
        """
        if len(pre_means) < 2:
            logger.warning("CUPED skipped: need at least 2 units with pre-means.")
            return synthetic, real, None
        variance_pre = np.var(pre_means, ddof=1)
        if np.isclose(variance_pre, 0.0):
            logger.warning("CUPED skipped: variance of pre_means is zero.")
            return synthetic, real, None
        covariance = np.cov(synthetic, pre_means)[0, 1]
        theta = float(covariance / variance_pre)
        adjusted_real = [r - theta * pre for r, pre in zip(real, pre_means)]
        adjusted_synthetic = [s - theta * pre for s, pre in zip(synthetic, pre_means)]
        return adjusted_synthetic, adjusted_real, theta

    @staticmethod
    def __run_inference(
        synthetic: List[float],
        real: List[float],
        inference_method: Literal["independent_ttest", "paired_ttest", "permutation"],
        permutation_iterations: int,
        random_state: int,
    ) -> tuple[str, float, float]:
        """Run statistical inference for experiment-level effect.

        Args:
            synthetic (List[float]): Synthetic values after filtering/adjustment.
            real (List[float]): Real values after filtering/adjustment.
            inference_method (Literal["independent_ttest", "paired_ttest", "permutation"]): Inference method.
            permutation_iterations (int): Iterations for permutation method.
            random_state (int): Random seed for permutation method.

        Returns:
            tuple[str, float, float]: Test name, statistic and p-value.
        """
        syn = np.asarray(synthetic, dtype=float)
        real_arr = np.asarray(real, dtype=float)
        if inference_method == "independent_ttest":
            ttest = stats.ttest_ind(syn, real_arr, equal_var=False, nan_policy="omit")
            return "independent_ttest", float(ttest.statistic), float(ttest.pvalue)
        if inference_method == "paired_ttest":
            if len(syn) != len(real_arr) or len(syn) < 2:
                raise ValueError(
                    "paired_ttest requires equal-length arrays with at least 2 observations."
                )
            ttest = stats.ttest_rel(syn, real_arr, nan_policy="omit")
            return "paired_ttest", float(ttest.statistic), float(ttest.pvalue)
        if inference_method == "permutation":
            if permutation_iterations <= 0:
                raise ValueError(
                    f"permutation_iterations should be positive, got {permutation_iterations}."
                )
            observed = float(np.mean(real_arr) - np.mean(syn))
            joined = np.concatenate([real_arr, syn])
            n_real = len(real_arr)
            rng = np.random.default_rng(random_state)
            count = 0
            for _ in range(permutation_iterations):
                perm = rng.permutation(joined)
                stat = float(np.mean(perm[:n_real]) - np.mean(perm[n_real:]))
                if abs(stat) >= abs(observed):
                    count += 1
            pvalue = (count + 1) / (permutation_iterations + 1)
            return "permutation", observed, float(pvalue)
        raise ValueError(
            f"Unknown inference_method={inference_method}. "
            "Use 'independent_ttest', 'paired_ttest', or 'permutation'."
        )

    def get_metrics(self) -> dict[str, Union[float, tuple[float, float], str]]:
        """Return summary metrics from the latest run.

        Returns:
            dict[str, Union[float, tuple[float, float], str]]: Summary metrics for monitoring/reporting.
        """
        if self.latest_result is None:
            raise ValueError("Run experiment first to obtain metrics.")

        metrics = {
            "mean_synthetic": self.latest_result.mean_synthetic,
            "mean_real": self.latest_result.mean_real,
            "expected_effect_in_percent": self.latest_result.expected_effect_in_percent,
            "ttest_statistic": self.latest_result.test_statistic,
            "ttest_pvalue": self.latest_result.test_pvalue,
            "test_name": self.latest_result.test_name,
            "included_units": len(self.latest_result.included_units),
            "excluded_units": len(self.latest_result.excluded_units),
        }
        if self.cuped_enabled:
            metrics["theta"] = self.latest_result.theta
        return metrics
