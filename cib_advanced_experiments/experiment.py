import datetime
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from tqdm import tqdm

from cib_advanced_experiments.synthetic_control import SyntheticControl
from cib_advanced_experiments.visualization import (
    create_experiment_distributions_figure,
)


class Experiment:
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
        """
        Based class for processing experiments with given data.

        Args:
            treatments_dataframe (pd.DataFrame): pandas DataFrame with history of treatment dates with columns:
                unit_column (str), treatment_date_column (str)

            treatment_date_column (str): Name of column in treatments_dataframe DataFrame, which responses to treatment date

            Look for other required fields docs in SyntheticControl class in synthetic_control module.
            They are absolutely similar and just passed directly to SyntheticControl class.

        Returns:
            None
        """

        self.data = data
        self.treatments_dataframe = treatments_dataframe
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

    def __validate_treatments_dataframe(self):
        """
        Just validates that given dataframe with treatment dates is OK.
        """
        treatments_dataframe_axis = self.treatments_dataframe.columns.values
        assert (
            self.treatment_date_column in treatments_dataframe_axis
        ), f"No {self.treatment_date_column} in {treatments_dataframe_axis=}"
        assert (
            self.unit_column in treatments_dataframe_axis
        ), f"No {self.unit_column} in {treatments_dataframe_axis=}"

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
        to_plot: bool = True,
        **kwargs,
    ) -> None:
        """
        Runs every unit from treatments_dataframe via SyntheticControl.
        Method to run experiment. Evaluates the given experiment run with T-test.

        Args:
            feature (str): Which feature to use as a target.
            top_weights_to_plot (int, optional): How many top high weights to show on plot. Defaults to 5. Just Passes directly to SyntheticControl.
            effect_evaluation_aggregation (Literal["last_date", "mean_value"], optional): Aggregate effect on the last date of prediction or the mean alongside the prediction range.
                Defaults to "mean_value". To see the effect on the last dot or mean across the test data
            quantile_error_border (float, optional): On which quantile to cut the fitting errors. Defaults to 0.95.
            number_of_units_for_matching (int, optional): How many units to match for control group for every treated unit. Just Passes directly to SyntheticControl.
            categorical_matching (List[str], optional): Features to use for exact matching (categorical).
            numeric_matching (List[str], optional): Features to use for numeric matching (numerical).
            to_plot (bool, optional): Plot each SyntheticControl result or not. Defaults to True.

        Raises:
            ValueError: Raised if wrong effect_evaluation_aggregation is passed, not from ["last_date", "mean_value"]

        Returns:
            None
        """

        assert feature in self.features, f"{feature=} should be from {self.features=}"

        possible_effect_evaluation_aggregations = ["last_date", "mean_value"]
        assert (
            effect_evaluation_aggregation in possible_effect_evaluation_aggregations
        ), f"{effect_evaluation_aggregation} should be from {possible_effect_evaluation_aggregations}"

        resulting_synthetic_controls = []
        for treated_unit in tqdm(
            self.treatments_dataframe[self.unit_column].unique(),
            desc=f"Synth for treated_unit",
        ):
            if not (treated_unit in self.data[self.unit_column].unique()):
                logger.info(
                    f"{treated_unit} not in history DataFrame. Skip fitting it."
                )
                continue
            assert (
                treated_unit in self.treatments_dataframe[self.unit_column].unique()
            ), f"{treated_unit} not in treatments_dataframe DataFrame. You are definitely doing something wrong."

            treatment_date = self.treatments_dataframe[
                self.treatments_dataframe[self.unit_column] == treated_unit
            ].iloc[0][self.treatment_date_column]

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
                date_frequency=self.date_frequency,
                aggregation_statistic=self.aggregation_statistic,
                **kwargs,
            )
            synthetic_control.fit()
            if to_plot:
                synthetic_control.plot_results(feature, top_weights_to_plot)

            resulting_synthetic_controls.append(synthetic_control)
        self.resulting_synthetic_controls = resulting_synthetic_controls

        real, synthetic, pre_means = [], [], []
        # taking quantile to filter bad fits
        fit_errors = [
            resulting_synthetic_control.model.get_loss_value()
            for resulting_synthetic_control in self.resulting_synthetic_controls
        ]
        border_quantile = float(
            round(np.quantile(fit_errors, quantile_error_border), 3)
        )
        logger.info("==========================================================")
        logger.info("====================EVALUATION RESULTS====================")
        logger.info("==========================================================")
        logger.info(
            f"Filtering {border_quantile=} with {quantile_error_border=}. If some unit from treatment_group fitted worse, do not take it into experiment evaluation:"
        )
        for resulting_synthetic_control in self.resulting_synthetic_controls:
            data = resulting_synthetic_control.get_data_on_features(feature)

            if effect_evaluation_aggregation == "last_date":
                data = data[data["date"] == data["date"].max()]
            elif effect_evaluation_aggregation == "mean_value":
                data = data[data["date"] >= resulting_synthetic_control.treatment_date]

            synth_val = data["synthetic"].mean()
            real_val = data["real"].mean()

            pre_mean = resulting_synthetic_control.get_pre_treatment_mean(feature)
            fit_error = resulting_synthetic_control.model.get_loss_value()

            if fit_error < border_quantile:
                real.append(real_val)
                synthetic.append(synth_val)
                pre_means.append(pre_mean)
            else:
                logger.warning(
                    f"Bad fit for unit {resulting_synthetic_control.treated_unit} with {resulting_synthetic_control.model.loss_func} error = {fit_error}"
                )

        logger.info("-" * 20)

        if self.cuped_enabled:
            if len(pre_means) == 0:
                error_msg = (
                    "Cuped is True but no valid pre-treatment data available. "
                    f"All {len(self.resulting_synthetic_controls)} units were filtered out "
                    f"due to poor fit quality (border_quantile={border_quantile})"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"Applying Cuped adjustment to {len(pre_means)} units")

            # Cuped adjustment logic
            covariance = np.cov(synthetic, pre_means)[0, 1]
            variance_pre = np.var(pre_means, ddof=1)
            if variance_pre == 0:
                logger.warning(
                    "Variance of pre_means is zero, Cuped adjustment skipped"
                )
                logger.info("Using standard evaluation without Cuped")
            else:
                theta = covariance / variance_pre
                logger.info(f"Cuped theta coefficient: {theta:.4f}")
                adjusted_real = [r - theta * pre for r, pre in zip(real, pre_means)]
                adjusted_synthetic = [
                    s - theta * pre for s, pre in zip(synthetic, pre_means)
                ]
                real, synthetic = adjusted_real, adjusted_synthetic
                logger.info("Cuped adjustment successfully applied")
        else:
            logger.info("Cuped is disabled, using standard evaluation")

        mean_synthetic, mean_real = np.mean(synthetic), np.mean(real)
        expected_effect_in_percent = round(
            (mean_real - mean_synthetic) / mean_synthetic * 100, 3
        )
        logger.info(
            f"Synthetic  {feature} '{effect_evaluation_aggregation}' aggregation mean: {mean_synthetic}",
        )
        logger.info(
            f"Real value {feature} '{effect_evaluation_aggregation}' aggregation mean: {mean_real}",
        )
        logger.info(f"Expected effect {expected_effect_in_percent}%")
        logger.info(
            f"Running t-test on {len(real)} fitted and available samples from given control group"
        )
        ttest = self.__get_ttest(synthetic, real)
        logger.info(f"T-tests result: {ttest=}")
        logger.info("-" * 20)

        fig = create_experiment_distributions_figure(
            real=real,
            synthetic=synthetic,
            fit_errors=fit_errors,
            border_quantile=border_quantile,
        )
        fig.show()

        self.mean_synthetic = mean_synthetic
        self.mean_real = mean_real
        self.expected_effect_in_percent = expected_effect_in_percent
        self.ttest = ttest
        if self.cuped_enabled:
            self.theta = theta
        else:
            self.theta = None

    def __get_ttest(
        self, synthetic: List[float], real: List[float]
    ) -> Union[float, float]:
        """Just counts T-test between synthetic and control arrays

        Args:
            synthetic (List[float]): Array of synthetic values
            real (List[float]): Array of real values

        Returns:
            _type_: scipy.stats t-test result
        """
        ttest = stats.ttest_ind(synthetic, real)
        return ttest

    def get_metrics(self) -> dict[str, Union[float, tuple[float, float]]]:
        metrics = {
            "mean_synthetic": self.mean_synthetic,
            "mean_real": self.mean_real,
            "expected_effect_in_percent": self.expected_effect_in_percent,
            "ttest_statistic": self.ttest.statistic,
            "ttest_pvalue": self.ttest.pvalue,
        }

        if self.cuped_enabled:
            metrics["theta"] = self.theta

        return metrics
