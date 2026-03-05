import datetime
import warnings
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from matplotlib import style

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
    create_synthetic_control_series_figure,
    create_synthetic_control_weights_figure,
)

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

style.use("fivethirtyeight")


class SyntheticControl:
    """Module for processing synthetic control.
    Realization is inspired by: https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html

    Args:
        data (pd.DataFrame): Dataframe on which the stynthetic control will be processed.
            Should have the the following column structure: unit_column, date_column, feat_0, feat_1,..., feat_n.
            Each row in dataset represents one observation.

        unit_column (str): Column from 'data' which divides units(cnum, cocunut, etc.).
        date_column (str): Column from 'data' with the date of event rows(trade_date, deal_date, etc.).
        features (str): Columns from 'data' with the numeric features we want to use for fitting
            and check the effect on(pnl, amount, etc.).

        treated_unit (Union[str, int]): Unit from the value of date[unit_column], which was treated and we want to
            see the effect on.

        treatment_date (Union[str, datetime.datetime], optional): Date of test unit been treated. Should be in the range of 'date_column' values
            or between 'min_date' and 'max_date'.

        control_group (List[Union[str, int]], optional): Group of ids from date[unit_column] which to be taken for control group.
            If None, the automactic matching would be done.

        min_date (Union[str, datetime.datetime], optional): Minimal date of column 'date_column' of dataset 'data'
            to take for synthetic control, including. Defaults to None. If None, the minimal date from 'date' will be taken.

        max_date (Union[str, datetime.datetime], optional): Maximal date of column 'date_column' of dataset 'data'
            to take for synthetic control, including. Defaults to None. If None, the maximal date from 'date' will be taken.

        date_frequency (Literal["D", "W", "M", "Y], optional): Data frequency aggregation based on time.
        Possible: "D": daily, "W": weekly, "M": monthly, "Y": yearly. Defaults to "D".

        aggregation_statistics (Literal["sum", "mean", "max", "min"], optional): Statistics of arggregation.
            For example, if we work with fx deals, and want to have data on mean deal_turnover per week, we set:
            date_frequency = "W", aggregation_statistics = "mean". Defaults to "mean".

        **seed (int): Random seed for model.
        **loss_func(Literal["mse", "mae"]): Optimize function for model.
        **epochs (int): Number of epochs for studying of the model.
        **lr (float): Learning rate of the model.

    Returns:
        None
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
        self.data = data.copy()
        # emperical + dont't touch it + don't care
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
        for column in [unit_column, date_column] + features:
            assert (
                column in self.data.columns.values
            ), f"Column {column} not in data axis"
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
        assert (
            str(treated_unit) in self.data[self.unit_column].values
        ), f"Unit {treated_unit} not in {self.unit_column} axis"
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
            assert (
                number_of_units_for_matching
            ), "Set 'number_of_units_for_matching' variable to some integer value, if you do not want to use matching set the 'control_group' directly."
            if matching_method == "feature":
                assert (
                    categorical_matching or numeric_matching
                ), "Set either 'categorical_matching' or 'numeric_matching' for feature matching."
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
        assert (
            len(control_group) >= self.MINIMAL_CONTROL_GROUP_SIZE
        ), f"To small 'control_group'. Make it at least {self.MINIMAL_CONTROL_GROUP_SIZE} units."
        # otherwise the given control group is taken directly
        # Given ids may be of not same type as ids in unit_column.
        # Convert to strings too for consistency.
        control_group = list(map(str, control_group))
        # if treated in control group, drop it
        control_group = [x for x in control_group if x != self.treated_unit]
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
        assert (treatment_date > self.min_date) and (
            treatment_date < self.max_date
        ), f"treatment_date should be between min_date and max_date. {treatment_date=}, {self.min_date=}, {self.max_date=}"
        self.treatment_date = treatment_date

    def __prepare_panel_and_matrices(self) -> None:
        """Build balanced panel and synthetic-control matrices with shared preprocessing."""
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
        """
        _validate_aggregation(aggregation_statistic)
        self.aggregation_statistic = aggregation_statistic

    def __filter_data_by_date(self) -> None:
        """
        We do not need all the data range, so we filter

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
        self.model = SynthControlModel(**kwargs)
        self.__weights = None

    def fit(self) -> None:
        """
        Method to fit synthetic control. Call after initialization.

        Returns:
            None
        """
        self.model.fit(self.__X, self.__y)
        self.__weights = self.model.get_weights()

    def get_weights(self) -> np.ndarray:
        """
        Get the weights of the synthetic control model.

        Returns:
            np.ndarray: Array of weights.
        """
        return self.__weights

    def get_data_on_features(self, feature: str) -> pd.DataFrame:
        """
        Get the real and synthetic data for a specific feature.

        Args:
            feature (str): Feature to get data for.

        Returns:
            pd.DataFrame: DataFrame with real and synthetic data for the feature.
        """
        assert feature in self.features, f"Select feature from {self.features}"
        assert self.__weights is not None, "Fit model first with .fit()"
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
        """
        treated_data = self.data[
            (self.data[self.unit_column] == self.treated_unit)
            & (self.data[self.date_column] < self.treatment_date)
        ]
        return treated_data[feature].mean()

    def plot_results(self, feature: str, top_weights_to_plot: int = 5) -> None:
        """
        Method plots the 'feature' of 'treated_unit' and synth of 'control_group'.
        Also plots the fitting weights distribution.

        Args:
            feature (str): Feature to plot from 'features'.
            top_weights_to_plot (int, optional): Number of top weights to plot. Defaults to 5.

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
