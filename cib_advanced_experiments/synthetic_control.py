"""High-level synthetic-control estimator and placebo inference.

The estimator was inspired by the synthetic-control chapters in Matheus
Facure's *Causal Inference for the Brave and True*. See
``THIRD_PARTY_NOTICES.md`` for attribution and licensing details.
"""

import datetime
from dataclasses import dataclass
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from cib_advanced_experiments.conformal_inference import (
    conformal_confidence_interval,
    conformal_p_value,
    conformal_p_value_grid_for_period,
    conformal_residuals,
)
from cib_advanced_experiments.matching import select_control_group_by_features
from cib_advanced_experiments.model import SynthControlModel
from cib_advanced_experiments.preprocessing import (
    _validate_aggregation,
    _validate_frequency,
    build_synthetic_control_matrices,
    prepare_panel_data,
)
from cib_advanced_experiments.propensity_score_matching import (
    select_control_group_by_propensity_from_panel,
)
from cib_advanced_experiments.visualization import (
    create_conformal_ci_band_figure,
    create_conformal_dashboard_figure,
    create_conformal_p_value_grid_figure,
    create_conformal_residuals_decomposition_figure,
    create_conformal_residuals_figure,
    create_placebo_distribution_figure,
    create_placebo_gap_figure,
    create_synthetic_control_effect_figure,
    create_synthetic_control_series_figure,
    create_synthetic_control_weights_figure,
)


@dataclass(frozen=True)
class PlaceboResult:
    """Container with placebo-series outputs and randomization-style inference.

    Attributes:
        placebo_effects (pd.DataFrame): Long frame with placebo/treated series and effects.
        unit_level_summary (pd.DataFrame): Per-unit placebo summary before filtering.
        filtered_unit_level_summary (pd.DataFrame): Per-unit summary after pre-fit filter.
        pre_error_border (float): Quantile threshold applied on pre-treatment MSE.
        treated_effect (float): Treated-unit aggregated effect value.
        placebo_pvalue (float): Placebo p-value against selected alternative.
        alternative (str): Alternative hypothesis used for p-value.
        effect_evaluation_aggregation (str): Aggregation mode used for unit-level effect.
    """

    placebo_effects: pd.DataFrame
    unit_level_summary: pd.DataFrame
    filtered_unit_level_summary: pd.DataFrame
    pre_error_border: float
    treated_effect: float
    placebo_pvalue: float
    alternative: str
    effect_evaluation_aggregation: str


class SyntheticControl:
    """Estimate a treated unit's counterfactual from a weighted donor pool.

    Args:
        data: Long-form panel or event dataframe.
        unit_column: Column identifying observational units.
        date_column: Column containing observation dates.
        features: Numeric outcomes used for fitting and evaluation.
        treated_unit: Identifier of the treated unit.
        treatment_date: First treated period.
        control_group: Explicit donor identifiers. If omitted, matching is used.
        min_date: Optional inclusive lower date bound.
        max_date: Optional inclusive upper date bound.
        number_of_units_for_matching: Number of controls selected automatically.
        categorical_matching: Exact-match columns for feature matching.
        numeric_matching: Numeric columns for distance-based matching.
        matching_method: ``"feature"`` or ``"propensity"``.
        propensity_covariates: Covariates for propensity-score matching.
        propensity_caliper: Maximum propensity-score distance.
        propensity_random_state: Random seed for the propensity model.
        date_frequency: Aggregation frequency: daily, weekly, monthly, or yearly.
        aggregation_statistic: Aggregation applied within unit-period cells.
        **kwargs: Parameters forwarded to :class:`SynthControlModel`, including
            ``seed``, ``loss_func``, ``max_iter``, and ``tolerance``.

    Notes:
        Causal interpretation depends on a credible donor pool, no treated-unit
        interference, and a sufficiently good pre-treatment fit.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        unit_column: str,
        date_column: str,
        features: List[str],
        treated_unit: Union[str, int],
        treatment_date: Union[str, datetime.datetime],
        control_group: List[Union[str, int]] = None,
        min_date: Union[str, datetime.datetime] = None,
        max_date: Union[str, datetime.datetime] = None,
        number_of_units_for_matching: int = None,
        categorical_matching: List[str] = None,
        numeric_matching: List[str] = None,
        matching_method: Literal["feature", "propensity"] = "feature",
        propensity_covariates: List[str] = None,
        propensity_caliper: float = None,
        propensity_random_state: int = 42,
        date_frequency: Literal["D", "W", "M", "Y"] = "D",
        aggregation_statistic: Literal["sum", "mean", "max", "min"] = "mean",
        **kwargs,
    ) -> None:
        """Initialize synthetic control object.

        Args:
            data (pd.DataFrame): Source panel/event dataframe.
            unit_column (str): Unit identifier column.
            date_column (str): Date column.
            features (List[str]): Features/outcomes used in synthetic control.
            treated_unit (Union[str, int]): Treated unit identifier.
            treatment_date (Union[str, datetime.datetime]): Treatment date.
            control_group (List[Union[str, int]], optional): Explicit control group.
            min_date (Union[str, datetime.datetime], optional): Lower date bound.
            max_date (Union[str, datetime.datetime], optional): Upper date bound.
            number_of_units_for_matching (int, optional): Number of controls to match automatically.
            categorical_matching (List[str], optional): Feature matching exact columns.
            numeric_matching (List[str], optional): Feature matching numeric columns.
            matching_method (Literal["feature", "propensity"], optional): Matching strategy.
            propensity_covariates (List[str], optional): Covariates for propensity matching.
            propensity_caliper (float, optional): Caliper for propensity matching.
            propensity_random_state (int, optional): Random seed for propensity model.
            date_frequency (Literal["D", "W", "M", "Y"], optional): Time aggregation frequency.
            aggregation_statistic (Literal["sum", "mean", "max", "min"], optional): Aggregation operation.
            **kwargs: Additional parameters for SynthControlModel.

        Returns:
            None
        """
        self.data = data.copy()
        # A minimum donor pool reduces accidental overfitting in auto-matching mode.
        self.MINIMAL_CONTROL_GROUP_SIZE = 10
        self.__validate_data(
            unit_column=unit_column, date_column=date_column, features=features
        )
        self.__check_if_treated_unit_is_in_dataframe(treated_unit=treated_unit)
        self.__init_data_frequency(date_frequency=date_frequency)
        self.__init_aggregation_statistic(aggregation_statistic=aggregation_statistic)
        self.__handle_dates(
            min_date=min_date, max_date=max_date, treatment_date=treatment_date
        )
        self.__filter_data_by_date()
        self.__set_control_group(
            control_group=control_group,
            number_of_units_for_matching=number_of_units_for_matching,
            categorical_matching=categorical_matching,
            numeric_matching=numeric_matching,
            matching_method=matching_method,
            propensity_covariates=propensity_covariates,
            propensity_caliper=propensity_caliper,
            propensity_random_state=propensity_random_state,
        )
        self.__prepare_panel_and_matrices()
        self.__aggregate_data()
        self.__init_model(kwargs)

    def __validate_data(
        self, unit_column: str, date_column: str, features: List[str]
    ) -> None:
        """
        Check that all given columns are in the given dataframe.

        Args:
            unit_column (str): Column name for units.
            date_column (str): Column name for dates.
            features (List[str]): List of feature column names.

        Returns:
            None
        """
        required_columns = [unit_column, date_column] + features
        missing_columns = [
            column for column in required_columns if column not in self.data.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}.")
        self.unit_column = unit_column
        # lets have ids as strings
        self.data[self.unit_column] = self.data[self.unit_column].astype(str)
        self.date_column = date_column
        self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
        self.features = features

    def __check_if_treated_unit_is_in_dataframe(
        self, treated_unit: Union[str, int]
    ) -> None:
        """
        Check that the treated unit id is in the dataset.

        Args:
            treated_unit (Union[str, int]): The treated unit id.

        Returns:
            None
        """
        if str(treated_unit) not in self.data[self.unit_column].values:
            raise ValueError(
                f"Treated unit {treated_unit!r} is not present in "
                f"column {self.unit_column!r}."
            )
        self.treated_unit = str(treated_unit)

    def __set_control_group(
        self,
        control_group: List[str],
        number_of_units_for_matching: Union[int, None] = None,
        categorical_matching: List[str] = None,
        numeric_matching: List[str] = None,
        matching_method: Literal["feature", "propensity"] = "feature",
        propensity_covariates: List[str] = None,
        propensity_caliper: float = None,
        propensity_random_state: int = 42,
    ) -> None:
        """
        Get only the given ids from the dataframe, if given.
        If they are not given, all units are taken.

        Args:
            control_group (List[str]): List of control unit ids.
            features (features: List[str]): On which features to make matching
            number_of_units_for_matching (int, optional): How many units to take into matching

        Returns:
            None
        """
        if control_group and number_of_units_for_matching:
            raise ValueError(
                """You passed both 'control_group' directly and 'number_of_units_for_matching'. Pass only one of them: 'control_group' 
            if you want to set your own or 'number_of_units_for_matching' if you want to match the group."""
            )
        if not control_group:
            # if control group is not set, make matching
            if number_of_units_for_matching is None:
                raise ValueError(
                    "Set 'number_of_units_for_matching' when 'control_group' "
                    "is not provided."
                )
            if matching_method == "feature":
                if not (categorical_matching or numeric_matching):
                    raise ValueError(
                        "Set either 'categorical_matching' or 'numeric_matching' "
                        "for feature matching."
                    )
                matching_result = select_control_group_by_features(
                    data=self.data,
                    unit_column=self.unit_column,
                    date_column=self.date_column,
                    treated_unit=self.treated_unit,
                    treatment_date=self.treatment_date,
                    number_of_units=number_of_units_for_matching,
                    categorical_features=categorical_matching,
                    numeric_features=numeric_matching,
                )
            elif matching_method == "propensity":
                matching_result = select_control_group_by_propensity_from_panel(
                    data=self.data,
                    unit_column=self.unit_column,
                    date_column=self.date_column,
                    treated_unit=self.treated_unit,
                    treatment_date=self.treatment_date,
                    covariates=propensity_covariates,
                    number_of_controls=number_of_units_for_matching,
                    caliper=propensity_caliper,
                    random_state=propensity_random_state,
                )
            else:
                raise ValueError(
                    f"Unknown matching_method='{matching_method}'. Use 'feature' or 'propensity'."
                )
            control_group = matching_result.control_group
            self.control_group_via_matching = True
        else:
            self.control_group_via_matching = False
        # Normalize identifiers, remove duplicates while preserving order, and
        # never allow the treated unit into its own donor pool.
        control_group = list(
            dict.fromkeys(
                str(unit) for unit in control_group if str(unit) != self.treated_unit
            )
        )
        available_units = set(self.data[self.unit_column].astype(str))
        missing_controls = [
            unit for unit in control_group if unit not in available_units
        ]
        if missing_controls:
            raise ValueError(
                f"Control units are not present in {self.unit_column!r}: "
                f"{missing_controls}."
            )
        if len(control_group) < self.MINIMAL_CONTROL_GROUP_SIZE:
            raise ValueError(
                "Control group is too small after validation: need at least "
                f"{self.MINIMAL_CONTROL_GROUP_SIZE} distinct control units, "
                f"got {len(control_group)}."
            )
        self.data = self.data[
            self.data[self.unit_column].isin(control_group + [self.treated_unit])
        ].copy()
        self.control_group = control_group

    def __handle_dates(
        self,
        min_date: Union[str, datetime.datetime],
        max_date: Union[str, datetime.datetime],
        treatment_date: Union[str, datetime.datetime],
    ) -> None:
        """
        Make similar format for all dates used.

        Args:
            min_date (Union[str, datetime.datetime], optional): Minimal date.
            max_date (Union[str, datetime.datetime], optional): Maximal date.
            treatment_date (Union[str, datetime.datetime], optional): Date of treatment.

        Returns:
            None
        """
        # If max_date/min_date are not given, we take max and min of dates range from 'date_column'
        self.min_date = pd.to_datetime(
            min_date if min_date else self.data[self.date_column].dt.date.min()
        )
        self.max_date = pd.to_datetime(
            max_date if max_date else self.data[self.date_column].dt.date.max()
        )
        treatment_date = pd.to_datetime(treatment_date)
        if not self.min_date < treatment_date < self.max_date:
            raise ValueError(
                "treatment_date should be strictly between min_date and max_date; "
                f"got treatment_date={treatment_date}, min_date={self.min_date}, "
                f"max_date={self.max_date}."
            )
        self.treatment_date = treatment_date

    def __prepare_panel_and_matrices(self) -> None:
        """Build balanced panel for downstream synthetic-control matrices.

        Returns:
            None
        """
        self.data = prepare_panel_data(
            data=self.data,
            unit_column=self.unit_column,
            date_column=self.date_column,
            value_columns=self.features,
            min_date=self.min_date,
            max_date=self.max_date,
            date_frequency=self.date_frequency,
            aggregation_statistic=self.aggregation_statistic,
            fill_value=0.0,
        )

    def __init_data_frequency(
        self, date_frequency: Literal["D", "W", "M", "Y"]
    ) -> None:
        """
        Function sets the aggregation granularity for the processed dataset.

        Args:
            date_frequency (Literal["D", "W", "M", "Y"]): Frequency of data aggregation.

        Returns:
            None
        """
        _validate_frequency(date_frequency)
        self.date_frequency = date_frequency

    def __init_aggregation_statistic(
        self, aggregation_statistic: Literal["sum", "mean", "max", "min"]
    ) -> None:
        """
        Function sets the aggregation statistic for the processed dataset.

        Args:
            aggregation_statistic (Literal["sum", "mean", "max", "min"]): Aggregation statistic.

        Returns:
            None
        """
        _validate_aggregation(aggregation_statistic)
        self.aggregation_statistic = aggregation_statistic

    def __filter_data_by_date(self) -> None:
        """
        We do not need all the data range, so we filter

        Args:
            None

        Returns:
            None
        """
        self.data = self.data[
            (self.data[self.date_column] >= self.min_date)
            & (self.data[self.date_column] <= self.max_date)
        ]

    def __aggregate_data(self) -> None:
        """
        Pivoting data in needed format.

        Args:
            None

        Returns:
            None
        """
        matrices = build_synthetic_control_matrices(
            panel_data=self.data,
            unit_column=self.unit_column,
            date_column=self.date_column,
            features=self.features,
            treated_unit=self.treated_unit,
            treatment_date=self.treatment_date,
        )
        self.data = matrices["prepared_data"]
        self.processed_data = matrices["processed_data"]
        self.all_dates = matrices["all_dates"]
        self.__X = matrices["X"]
        self.__y = matrices["y"]

    def __init_model(self, kwargs):
        """
        Initialize the model with parameters from the constructor.

        Args:
            kwargs (dict): Additional keyword arguments for the SynthControlModel.

        Returns:
            None
        """
        self.model_kwargs = kwargs.copy()
        self.model = SynthControlModel(**kwargs)
        self.__weights = None

    def fit(self) -> None:
        """
        Method to fit synthetic control. Call after initialization.

        Args:
            None

        Returns:
            None
        """
        self.model.fit(self.__X, self.__y)
        self.__weights = self.model.get_weights()

    def get_weights(self) -> np.ndarray:
        """
        Get the weights of the synthetic control model.

        Args:
            None

        Returns:
            np.ndarray: Array of weights.
        """
        if self.__weights is None:
            raise ValueError("Fit model first with .fit()")
        return self.__weights

    def get_data_on_features(self, feature: str) -> pd.DataFrame:
        """
        Get the real and synthetic data for a specific feature.

        Args:
            feature (str): Feature to get data for.

        Returns:
            pd.DataFrame: DataFrame with real and synthetic data for the feature.
        """
        if feature not in self.features:
            raise ValueError(f"Select feature from {self.features}")
        if self.__weights is None:
            raise ValueError("Fit model first with .fit()")
        # put all values in atributed to be able to get them
        real = self.data.query("is_treated")[feature].to_list()
        synth = (
            self.data.query("~is_treated")
            .pivot(index=self.date_column, columns=self.unit_column)[feature]
            .astype(float)
            .values.dot(self.__weights)
        )
        result = pd.DataFrame(
            data={"date": self.all_dates, "real": real, "synthetic": synth}
        )
        return result

    def get_pre_treatment_mean(self, feature: str) -> float:
        """
        Mean value of feature for treated_unit in treatment_date.

        Args:
            feature (str): Target feature name.

        Returns:
            float: Mean pre-treatment feature value for treated unit.
        """
        treated_data = self.data[
            (self.data[self.unit_column] == self.treated_unit)
            & (self.data[self.date_column] < self.treatment_date)
        ]
        return treated_data[feature].mean()

    def plot_results(
        self, feature: str, top_weights_to_plot: int = 5, with_effect: bool = False
    ) -> None:
        """
        Method plots the 'feature' of 'treated_unit' and synth of 'control_group'.
        Also plots the fitting weights distribution.

        Args:
            feature (str): Feature to plot from 'features'.
            top_weights_to_plot (int, optional): Number of top weights to plot. Defaults to 5.
            with_effect (bool, optional): If True, also show effect/gap plot (real - synthetic).

        Returns:
            None
        """
        df_to_plot = self.get_data_on_features(feature)

        fig1 = create_synthetic_control_series_figure(
            df_to_plot=df_to_plot,
            treated_unit=self.treated_unit,
            treatment_date=self.treatment_date,
            feature=feature,
            control_group_via_matching=self.control_group_via_matching,
            loss_func=self.model.loss_func,
            fit_error=self.model.get_loss_value(),
        )

        weights = self.model.get_weights()
        fig2 = create_synthetic_control_weights_figure(
            weights=weights,
            top_weights_to_plot=top_weights_to_plot,
        )

        fig1.show()
        fig2.show()
        if with_effect:
            self.plot_effect(feature=feature)

    def get_effect_data(self, feature: str) -> pd.DataFrame:
        """Get effect/gap time series as real minus synthetic.

        Args:
            feature (str): Target feature name.

        Returns:
            pd.DataFrame: Dataframe with date, real, synthetic and effect columns.
        """
        data = self.get_data_on_features(feature=feature).copy()
        data["effect"] = data["real"] - data["synthetic"]
        return data

    def get_placebo_effects(
        self,
        feature: str,
        include_treated_unit: bool = True,
        raise_on_error: bool = True,
    ) -> pd.DataFrame:
        """Compute placebo effect paths by treating each unit as if treated.

        Args:
            feature (str): Target feature name.
            include_treated_unit (bool, optional): Include actual treated-unit fit in output.
            raise_on_error (bool, optional): Raise when one placebo fit fails.

        Returns:
            pd.DataFrame: Long dataframe with treated_unit/date/real/synthetic/effect.
        """
        if feature not in self.features:
            raise ValueError(f"Select feature from {self.features}")
        all_units = sorted(self.data[self.unit_column].astype(str).unique().tolist())
        if len(all_units) <= self.MINIMAL_CONTROL_GROUP_SIZE:
            raise ValueError(
                "Not enough units for placebo analysis: need at least "
                f"{self.MINIMAL_CONTROL_GROUP_SIZE + 1} total units, got {len(all_units)}."
            )

        placebo_frames = []
        for unit in tqdm(
            all_units,
            desc="Placebo effect",
        ):
            if not include_treated_unit and unit == self.treated_unit:
                continue

            donor_pool = [candidate for candidate in all_units if candidate != unit]
            try:
                placebo_sc = SyntheticControl(
                    data=self.data,
                    unit_column=self.unit_column,
                    date_column=self.date_column,
                    features=self.features,
                    treated_unit=unit,
                    treatment_date=self.treatment_date,
                    control_group=donor_pool,
                    min_date=self.min_date,
                    max_date=self.max_date,
                    date_frequency=self.date_frequency,
                    aggregation_statistic=self.aggregation_statistic,
                    **self.model_kwargs,
                )
                placebo_sc.fit()
                effect_data = placebo_sc.get_effect_data(feature=feature).copy()
                effect_data["treated_unit"] = str(unit)
                effect_data["is_actual_treated"] = str(unit) == self.treated_unit
                placebo_frames.append(effect_data)
            except Exception:
                if raise_on_error:
                    raise
                continue

        if len(placebo_frames) == 0:
            raise ValueError("No placebo series were computed.")
        return pd.concat(placebo_frames, ignore_index=True)

    @staticmethod
    def _aggregate_effect_series(
        effects_df: pd.DataFrame,
        treatment_date: pd.Timestamp,
        effect_evaluation_aggregation: Literal[
            "last_date", "mean_value"
        ] = "mean_value",
    ) -> float:
        """Aggregate one unit-level effect series into scalar summary."""
        if effect_evaluation_aggregation == "last_date":
            return float(
                effects_df.loc[
                    effects_df["date"] == effects_df["date"].max(), "effect"
                ].mean()
            )
        if effect_evaluation_aggregation == "mean_value":
            return float(
                effects_df.loc[effects_df["date"] >= treatment_date, "effect"].mean()
            )
        raise ValueError(
            "effect_evaluation_aggregation should be one of ['last_date', 'mean_value']"
        )

    def get_placebo_result(
        self,
        feature: str,
        effect_evaluation_aggregation: Literal[
            "last_date", "mean_value"
        ] = "mean_value",
        pre_error_quantile: float = 1.0,
        alternative: Literal["less", "greater", "two-sided"] = "less",
        raise_on_error: bool = True,
    ) -> PlaceboResult:
        """Compute placebo summary table and p-value in randomization style.

        Args:
            feature (str): Target feature name.
            effect_evaluation_aggregation (Literal["last_date", "mean_value"], optional): Aggregation for unit effects.
            pre_error_quantile (float, optional): Quantile filter on pre-treatment MSE in [0, 1].
            alternative (Literal["less", "greater", "two-sided"], optional): Alternative hypothesis.
            raise_on_error (bool, optional): Raise when one placebo fit fails.

        Returns:
            PlaceboResult: Placebo series, summaries and p-value.
        """
        if not (0.0 <= pre_error_quantile <= 1.0):
            raise ValueError(
                f"pre_error_quantile should be in [0,1], got {pre_error_quantile}."
            )
        if alternative not in {"less", "greater", "two-sided"}:
            raise ValueError(
                f"alternative should be one of ['less', 'greater', 'two-sided']; got {alternative}."
            )

        placebo_effects = self.get_placebo_effects(
            feature=feature,
            include_treated_unit=True,
            raise_on_error=raise_on_error,
        )
        placebo_effects["date"] = pd.to_datetime(placebo_effects["date"])
        treatment_date = pd.to_datetime(self.treatment_date)

        summary_rows = []
        for unit in sorted(
            placebo_effects["treated_unit"].astype(str).unique().tolist()
        ):
            unit_effects = placebo_effects[
                placebo_effects["treated_unit"] == unit
            ].copy()
            pre_block = unit_effects[unit_effects["date"] < treatment_date]
            if pre_block.empty:
                raise ValueError(f"No pre-treatment rows for placebo unit '{unit}'.")
            pre_mse = float(
                np.mean(np.square(pre_block["effect"].astype(float).values))
            )
            aggregated_effect = self._aggregate_effect_series(
                effects_df=unit_effects,
                treatment_date=treatment_date,
                effect_evaluation_aggregation=effect_evaluation_aggregation,
            )
            summary_rows.append(
                {
                    "treated_unit": str(unit),
                    "is_actual_treated": str(unit) == self.treated_unit,
                    "pre_treatment_mse": pre_mse,
                    "effect": aggregated_effect,
                }
            )

        unit_level_summary = pd.DataFrame(summary_rows).sort_values("treated_unit")
        placebo_only = unit_level_summary[~unit_level_summary["is_actual_treated"]]
        pre_error_border = float(
            np.quantile(placebo_only["pre_treatment_mse"], pre_error_quantile)
        )

        filtered_unit_level_summary = unit_level_summary[
            (unit_level_summary["pre_treatment_mse"] <= pre_error_border)
            | (unit_level_summary["is_actual_treated"])
        ].copy()

        treated_row = filtered_unit_level_summary[
            filtered_unit_level_summary["is_actual_treated"]
        ]
        if treated_row.empty:
            raise ValueError("Treated unit is missing from placebo summary.")
        treated_effect = float(treated_row.iloc[0]["effect"])

        reference_effects = filtered_unit_level_summary["effect"].astype(float).values
        if alternative == "less":
            placebo_pvalue = float(np.mean(reference_effects <= treated_effect))
        elif alternative == "greater":
            placebo_pvalue = float(np.mean(reference_effects >= treated_effect))
        else:
            placebo_pvalue = float(
                np.mean(np.abs(reference_effects) >= abs(treated_effect))
            )

        return PlaceboResult(
            placebo_effects=placebo_effects,
            unit_level_summary=unit_level_summary,
            filtered_unit_level_summary=filtered_unit_level_summary,
            pre_error_border=pre_error_border,
            treated_effect=treated_effect,
            placebo_pvalue=placebo_pvalue,
            alternative=alternative,
            effect_evaluation_aggregation=effect_evaluation_aggregation,
        )

    def plot_placebo_effects(
        self,
        feature: str,
        pre_error_quantile: float = 1.0,
        raise_on_error: bool = True,
    ) -> None:
        """Plot placebo effect paths with highlighted treated-unit path.

        Args:
            feature (str): Target feature name.
            pre_error_quantile (float, optional): Quantile filter on pre-treatment MSE in [0, 1].
            raise_on_error (bool, optional): Raise when one placebo fit fails.

        Returns:
            None
        """
        placebo_result = self.get_placebo_result(
            feature=feature,
            effect_evaluation_aggregation="mean_value",
            pre_error_quantile=pre_error_quantile,
            alternative="two-sided",
            raise_on_error=raise_on_error,
        )
        kept_units = set(
            placebo_result.filtered_unit_level_summary["treated_unit"]
            .astype(str)
            .tolist()
        )
        effects_to_plot = placebo_result.placebo_effects[
            placebo_result.placebo_effects["treated_unit"].astype(str).isin(kept_units)
        ].copy()
        fig = create_placebo_gap_figure(
            placebo_effects=effects_to_plot,
            treatment_date=self.treatment_date,
            actual_treated_unit=self.treated_unit,
        )
        fig.show()

    def plot_placebo_distribution(
        self,
        feature: str,
        effect_evaluation_aggregation: Literal[
            "last_date", "mean_value"
        ] = "mean_value",
        pre_error_quantile: float = 1.0,
        alternative: Literal["less", "greater", "two-sided"] = "less",
        raise_on_error: bool = True,
    ) -> None:
        """Plot placebo distribution of unit-level effects with treated marker.

        Args:
            feature (str): Target feature name.
            effect_evaluation_aggregation (Literal["last_date", "mean_value"], optional): Aggregation for unit effects.
            pre_error_quantile (float, optional): Quantile filter on pre-treatment MSE in [0, 1].
            alternative (Literal["less", "greater", "two-sided"], optional): Alternative hypothesis.
            raise_on_error (bool, optional): Raise when one placebo fit fails.

        Returns:
            None
        """
        placebo_result = self.get_placebo_result(
            feature=feature,
            effect_evaluation_aggregation=effect_evaluation_aggregation,
            pre_error_quantile=pre_error_quantile,
            alternative=alternative,
            raise_on_error=raise_on_error,
        )
        fig = create_placebo_distribution_figure(
            effect_values=placebo_result.filtered_unit_level_summary["effect"]
            .astype(float)
            .tolist(),
            treated_effect=placebo_result.treated_effect,
            effect_evaluation_aggregation=effect_evaluation_aggregation,
        )
        fig.show()

    def plot_effect(self, feature: str) -> None:
        """Plot estimated treatment effect (gap) over time.

        Args:
            feature (str): Target feature name.

        Returns:
            None
        """
        df_to_plot = self.get_data_on_features(feature=feature)
        fig = create_synthetic_control_effect_figure(
            df_to_plot=df_to_plot,
            treated_unit=self.treated_unit,
            treatment_date=self.treatment_date,
            feature=feature,
        )
        fig.show()

    def get_conformal_residuals(
        self,
        feature: str,
        null_hypothesis: Union[float, int, np.ndarray, list[float]] = 0.0,
        intervention_start: Union[str, datetime.datetime, pd.Timestamp] = None,
        window: int = None,
    ) -> pd.DataFrame:
        """Compute conformal residuals under a null hypothesis.

        Args:
            feature (str): Target feature name.
            null_hypothesis (Union[float, int, np.ndarray, list[float]], optional): Null effect.
            intervention_start (Union[str, datetime.datetime, pd.Timestamp], optional): Intervention date.
            window (int, optional): Number of post periods to include.

        Returns:
            pd.DataFrame: Residual dataframe for conformal inference.
        """
        if feature not in self.features:
            raise ValueError(f"Select feature from {self.features}")
        return conformal_residuals(
            panel_data=self.data,
            unit_column=self.unit_column,
            date_column=self.date_column,
            outcome_column=feature,
            treated_unit=self.treated_unit,
            intervention_start=intervention_start or self.treatment_date,
            null_hypothesis=null_hypothesis,
            model_kwargs=self.model_kwargs,
            window=window,
        )

    def get_conformal_p_value(
        self,
        feature: str,
        null_hypothesis: Union[float, int, np.ndarray, list[float]] = 0.0,
        intervention_start: Union[str, datetime.datetime, pd.Timestamp] = None,
        window: int = None,
        q: int = 1,
    ) -> float:
        """Compute permutation conformal p-value.

        Args:
            feature (str): Target feature name.
            null_hypothesis (Union[float, int, np.ndarray, list[float]], optional): Null effect.
            intervention_start (Union[str, datetime.datetime, pd.Timestamp], optional): Intervention date.
            window (int, optional): Number of post periods to include.
            q (int, optional): Norm parameter for test statistic.

        Returns:
            float: Conformal p-value.
        """
        if feature not in self.features:
            raise ValueError(f"Select feature from {self.features}")
        return conformal_p_value(
            panel_data=self.data,
            unit_column=self.unit_column,
            date_column=self.date_column,
            outcome_column=feature,
            treated_unit=self.treated_unit,
            intervention_start=intervention_start or self.treatment_date,
            null_hypothesis=null_hypothesis,
            model_kwargs=self.model_kwargs,
            window=window,
            q=q,
        )

    def get_conformal_p_value_grid_for_period(
        self,
        feature: str,
        period: Union[str, datetime.datetime, pd.Timestamp],
        nulls: Union[np.ndarray, list[float]],
        intervention_start: Union[str, datetime.datetime, pd.Timestamp] = None,
        q: int = 1,
    ) -> pd.DataFrame:
        """Compute p-value grid over null effects for one post-treatment period.

        Args:
            feature (str): Target feature name.
            period (Union[str, datetime.datetime, pd.Timestamp]): Post-treatment period.
            nulls (Union[np.ndarray, list[float]]): Candidate null effects.
            intervention_start (Union[str, datetime.datetime, pd.Timestamp], optional): Intervention date.
            q (int, optional): Norm parameter for test statistic.

        Returns:
            pd.DataFrame: P-value grid indexed by null effects.
        """
        if feature not in self.features:
            raise ValueError(f"Select feature from {self.features}")
        return conformal_p_value_grid_for_period(
            panel_data=self.data,
            unit_column=self.unit_column,
            date_column=self.date_column,
            outcome_column=feature,
            treated_unit=self.treated_unit,
            intervention_start=intervention_start or self.treatment_date,
            period=period,
            nulls=nulls,
            model_kwargs=self.model_kwargs,
            q=q,
        )

    def get_conformal_confidence_interval(
        self,
        feature: str,
        nulls: Union[np.ndarray, list[float]],
        window: int,
        intervention_start: Union[str, datetime.datetime, pd.Timestamp] = None,
        alpha: float = 0.1,
        q: int = 1,
    ) -> pd.DataFrame:
        """Compute conformal confidence intervals by post-treatment period.

        Args:
            feature (str): Target feature name.
            nulls (Union[np.ndarray, list[float]]): Candidate null effects.
            window (int): Number of post periods.
            intervention_start (Union[str, datetime.datetime, pd.Timestamp], optional): Intervention date.
            alpha (float, optional): Significance level.
            q (int, optional): Norm parameter for test statistic.

        Returns:
            pd.DataFrame: Confidence interval bounds by period.
        """
        if feature not in self.features:
            raise ValueError(f"Select feature from {self.features}")
        return conformal_confidence_interval(
            panel_data=self.data,
            unit_column=self.unit_column,
            date_column=self.date_column,
            outcome_column=feature,
            treated_unit=self.treated_unit,
            intervention_start=intervention_start or self.treatment_date,
            window=window,
            nulls=nulls,
            model_kwargs=self.model_kwargs,
            alpha=alpha,
            q=q,
        )

    def plot_conformal_residuals(
        self,
        feature: str,
        null_hypothesis: Union[float, int, np.ndarray, list[float]] = 0.0,
        intervention_start: Union[str, datetime.datetime, pd.Timestamp] = None,
        window: int = None,
    ) -> None:
        """Plot conformal residual trajectory.

        Args:
            feature (str): Target feature name.
            null_hypothesis (Union[float, int, np.ndarray, list[float]], optional): Null effect.
            intervention_start (Union[str, datetime.datetime, pd.Timestamp], optional): Intervention date.
            window (int, optional): Number of post periods.

        Returns:
            None
        """
        residuals = self.get_conformal_residuals(
            feature=feature,
            null_hypothesis=null_hypothesis,
            intervention_start=intervention_start,
            window=window,
        )
        start = intervention_start or self.treatment_date
        fig = create_conformal_residuals_figure(
            residuals_df=residuals,
            intervention_start=pd.to_datetime(start),
        )
        fig.show()

    def plot_conformal_residuals_decomposition(
        self,
        feature: str,
        null_hypothesis: Union[float, int, np.ndarray, list[float]] = 0.0,
        intervention_start: Union[str, datetime.datetime, pd.Timestamp] = None,
        window: int = None,
    ) -> None:
        """Plot y0 vs y0_est and residuals in a side-by-side decomposition view.

        Args:
            feature (str): Target feature name.
            null_hypothesis (Union[float, int, np.ndarray, list[float]], optional): Null effect.
            intervention_start (Union[str, datetime.datetime, pd.Timestamp], optional): Intervention date.
            window (int, optional): Number of post periods.

        Returns:
            None
        """
        residuals = self.get_conformal_residuals(
            feature=feature,
            null_hypothesis=null_hypothesis,
            intervention_start=intervention_start,
            window=window,
        )
        start = intervention_start or self.treatment_date
        fig = create_conformal_residuals_decomposition_figure(
            residuals_df=residuals,
            intervention_start=pd.to_datetime(start),
        )
        fig.show()

    def plot_conformal_p_value_grid_for_period(
        self,
        feature: str,
        period: Union[str, datetime.datetime, pd.Timestamp],
        nulls: Union[np.ndarray, list[float]],
        intervention_start: Union[str, datetime.datetime, pd.Timestamp] = None,
        alpha: float = 0.1,
        q: int = 1,
    ) -> None:
        """Plot p-value grid over candidate null effects.

        Args:
            feature (str): Target feature name.
            period (Union[str, datetime.datetime, pd.Timestamp]): Post-treatment period.
            nulls (Union[np.ndarray, list[float]]): Candidate null effects.
            intervention_start (Union[str, datetime.datetime, pd.Timestamp], optional): Intervention date.
            alpha (float, optional): Significance level line.
            q (int, optional): Norm parameter for test statistic.

        Returns:
            None
        """
        p_values = self.get_conformal_p_value_grid_for_period(
            feature=feature,
            period=period,
            nulls=nulls,
            intervention_start=intervention_start,
            q=q,
        )
        fig = create_conformal_p_value_grid_figure(p_values_df=p_values, alpha=alpha)
        fig.show()

    def plot_conformal_confidence_interval(
        self,
        feature: str,
        nulls: Union[np.ndarray, list[float]],
        window: int,
        intervention_start: Union[str, datetime.datetime, pd.Timestamp] = None,
        alpha: float = 0.1,
        q: int = 1,
    ) -> None:
        """Plot estimated effect with conformal confidence interval band.

        Args:
            feature (str): Target feature name.
            nulls (Union[np.ndarray, list[float]]): Candidate null effects.
            window (int): Number of post periods.
            intervention_start (Union[str, datetime.datetime, pd.Timestamp], optional): Intervention date.
            alpha (float, optional): Significance level.
            q (int, optional): Norm parameter for test statistic.

        Returns:
            None
        """
        ci = self.get_conformal_confidence_interval(
            feature=feature,
            nulls=nulls,
            window=window,
            intervention_start=intervention_start,
            alpha=alpha,
            q=q,
        )
        data = self.get_data_on_features(feature=feature).copy()
        data["effect"] = data["real"] - data["synthetic"]
        start = pd.to_datetime(intervention_start or self.treatment_date)
        # Keep full effect timeline (as in article); CI is shown only for selected
        # post-treatment window via ci_df index.
        effect_df = data.copy()
        fig = create_conformal_ci_band_figure(
            effect_df=effect_df,
            ci_df=ci,
            treatment_date=start,
            alpha=alpha,
        )
        fig.show()

    def plot_conformal_dashboard(
        self,
        feature: str,
        nulls: Union[np.ndarray, list[float]],
        window: int,
        period: Union[str, datetime.datetime, pd.Timestamp] = None,
        intervention_start: Union[str, datetime.datetime, pd.Timestamp] = None,
        alpha: float = 0.1,
        q: int = 1,
    ) -> None:
        """Plot unified conformal dashboard with residuals, p-grid and CI band.

        Args:
            feature (str): Target feature name.
            nulls (Union[np.ndarray, list[float]]): Candidate null effects.
            window (int): Number of post periods.
            period (Union[str, datetime.datetime, pd.Timestamp], optional): Period for p-value grid.
            intervention_start (Union[str, datetime.datetime, pd.Timestamp], optional): Intervention date.
            alpha (float, optional): Significance level.
            q (int, optional): Norm parameter for test statistic.

        Returns:
            None
        """
        start = pd.to_datetime(intervention_start or self.treatment_date)
        data = self.get_data_on_features(feature=feature).copy()
        data["effect"] = data["real"] - data["synthetic"]
        post_dates = data.loc[data["date"] >= start, "date"].tolist()
        if len(post_dates) == 0:
            raise ValueError(
                "No post-treatment dates available for conformal dashboard."
            )
        selected_period = (
            pd.to_datetime(period) if period is not None else post_dates[0]
        )

        residuals = self.get_conformal_residuals(
            feature=feature,
            null_hypothesis=0.0,
            intervention_start=start,
            window=window,
        )
        p_values = self.get_conformal_p_value_grid_for_period(
            feature=feature,
            period=selected_period,
            nulls=nulls,
            intervention_start=start,
            q=q,
        )
        ci = self.get_conformal_confidence_interval(
            feature=feature,
            nulls=nulls,
            window=window,
            intervention_start=start,
            alpha=alpha,
            q=q,
        )
        # Keep full effect timeline (as in article); CI remains post-window only.
        effect_df = data.copy()

        fig = create_conformal_dashboard_figure(
            residuals_df=residuals,
            p_values_df=p_values,
            effect_df=effect_df,
            ci_df=ci,
            treatment_date=start,
            alpha=alpha,
        )
        fig.show()
