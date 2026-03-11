import datetime
from dataclasses import dataclass
from typing import List, Literal, Union

import numpy as np
import pandas as pd

DateLike = Union[str, datetime.datetime, pd.Timestamp]
AggregationStatistic = Literal["sum", "mean", "max", "min"]
DateFrequency = Literal["D", "W", "M", "Y"]


@dataclass(frozen=True)
class ConformalFrame:
    """Prepared panel blocks for conformal inference on Synthetic Control.

    Attributes:
        pre_treatment_treated (np.ndarray): Treated-unit outcomes before treatment.
        post_treatment_treated (np.ndarray): Treated-unit outcomes after treatment.
        pre_treatment_controls (pd.DataFrame): Control outcomes before treatment in wide form.
        post_treatment_controls (pd.DataFrame): Control outcomes after treatment in wide form.
        panel (pd.DataFrame): Original prepared panel dataframe.
    """

    pre_treatment_treated: np.ndarray
    post_treatment_treated: np.ndarray
    pre_treatment_controls: pd.DataFrame
    post_treatment_controls: pd.DataFrame
    panel: pd.DataFrame

# ====================== Synthetic Control ======================


def _ensure_columns(data: pd.DataFrame, columns: List[str]) -> None:
    """Raise ValueError if required columns are missing.

    Args:
        data (pd.DataFrame): Input dataframe.
        columns (List[str]): Required column names.

    Returns:
        None
    """
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_frequency(date_frequency: DateFrequency) -> None:
    """Validate allowed panel resampling frequency.

    Args:
        date_frequency (DateFrequency): Requested resampling frequency.

    Returns:
        None
    """
    possible_frequencies = {"D", "W", "M", "YE"}
    if date_frequency not in possible_frequencies:
        raise ValueError(
            f"Select date_frequency from {sorted(possible_frequencies)}; got '{date_frequency}'."
        )


def _validate_aggregation(aggregation_statistic: AggregationStatistic) -> None:
    """Validate allowed aggregation statistic.

    Args:
        aggregation_statistic (AggregationStatistic): Aggregation operation.

    Returns:
        None
    """
    possible_statistics = {"sum", "mean", "max", "min"}
    if aggregation_statistic not in possible_statistics:
        raise ValueError(
            f"Select aggregation_statistic from {sorted(possible_statistics)}; got '{aggregation_statistic}'."
        )


def _resolve_date_range(
    data: pd.DataFrame,
    date_column: str,
    min_date: DateLike = None,
    max_date: DateLike = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve and validate min/max date boundaries.

    Args:
        data (pd.DataFrame): Input dataframe with date column.
        date_column (str): Date column name.
        min_date (DateLike, optional): Lower bound date.
        max_date (DateLike, optional): Upper bound date.

    Returns:
        tuple[pd.Timestamp, pd.Timestamp]: Resolved min and max timestamps.
    """
    resolved_min = pd.to_datetime(min_date) if min_date else data[date_column].min()
    resolved_max = pd.to_datetime(max_date) if max_date else data[date_column].max()
    if resolved_min >= resolved_max:
        raise ValueError(
            f"min_date should be less than max_date, got {resolved_min} >= {resolved_max}."
        )
    return resolved_min, resolved_max


def prepare_panel_data(
    data: pd.DataFrame,
    unit_column: str,
    date_column: str,
    value_columns: List[str],
    min_date: DateLike = None,
    max_date: DateLike = None,
    date_frequency: DateFrequency = "D",
    aggregation_statistic: AggregationStatistic = "mean",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Build a balanced panel with one row per unit/date.

    Args:
        data (pd.DataFrame): Source dataframe.
        unit_column (str): Unit identifier column.
        date_column (str): Date column.
        value_columns (List[str]): Value columns to aggregate.
        min_date (DateLike, optional): Lower bound date.
        max_date (DateLike, optional): Upper bound date.
        date_frequency (DateFrequency, optional): Resampling frequency.
        aggregation_statistic (AggregationStatistic, optional): Aggregation operation.
        fill_value (float, optional): Fill value for missing unit/date points.

    Returns:
        pd.DataFrame: Balanced panel dataframe.
    """
    _ensure_columns(data, [unit_column, date_column] + value_columns)
    _validate_frequency(date_frequency)
    _validate_aggregation(aggregation_statistic)

    prepared = data.copy()
    prepared[unit_column] = prepared[unit_column].astype(str)
    prepared[date_column] = pd.to_datetime(prepared[date_column])

    min_date_ts, max_date_ts = _resolve_date_range(
        prepared, date_column=date_column, min_date=min_date, max_date=max_date
    )
    prepared = prepared[
        (prepared[date_column] >= min_date_ts) & (prepared[date_column] <= max_date_ts)
    ].copy()

    aggregations = {column: aggregation_statistic for column in value_columns}
    aggregated = (
        prepared.groupby(
            [unit_column, pd.Grouper(key=date_column, freq=date_frequency)],
            as_index=False,
        )
        .agg(aggregations)
        .sort_values([unit_column, date_column])
    )

    all_units = aggregated[unit_column].drop_duplicates().tolist()
    all_dates = pd.date_range(min_date_ts, max_date_ts, freq=date_frequency)
    full_idx = pd.MultiIndex.from_product(
        (all_units, all_dates), names=[unit_column, date_column]
    )
    balanced = (
        aggregated.set_index([unit_column, date_column])
        .reindex(full_idx, fill_value=fill_value)
        .reset_index()
        .sort_values([unit_column, date_column])
    )
    return balanced


def build_synthetic_control_matrices(
    panel_data: pd.DataFrame,
    unit_column: str,
    date_column: str,
    features: List[str],
    treated_unit: Union[str, int],
    treatment_date: DateLike,
) -> dict[str, Union[np.ndarray, pd.DataFrame, List[str], pd.Series]]:
    """Create matrices and helper blocks for synthetic control fitting.

    Args:
        panel_data (pd.DataFrame): Balanced panel dataframe.
        unit_column (str): Unit identifier column.
        date_column (str): Date column.
        features (List[str]): Outcome/features used in model.
        treated_unit (Union[str, int]): Treated unit identifier.
        treatment_date (DateLike): Treatment date.

    Returns:
        dict[str, Union[np.ndarray, pd.DataFrame, List[str], pd.Series]]: Prepared blocks.
    """
    _ensure_columns(panel_data, [unit_column, date_column] + features)
    treated_unit = str(treated_unit)
    treatment_date = pd.to_datetime(treatment_date)

    if treated_unit not in panel_data[unit_column].astype(str).unique():
        raise ValueError(f"Treated unit '{treated_unit}' is not present in panel_data.")

    prepared = panel_data.copy()
    prepared[unit_column] = prepared[unit_column].astype(str)
    prepared[date_column] = pd.to_datetime(prepared[date_column])
    prepared = prepared.sort_values(date_column)
    prepared["is_treated"] = prepared[unit_column] == treated_unit
    prepared["after_treatment"] = prepared[date_column] > treatment_date

    pre_treatment = prepared.query("~after_treatment")
    matrix = (
        pre_treatment.pivot(index=unit_column, columns=date_column)[features]
        .T.fillna(0.0)
    )
    y = matrix[treated_unit].values
    X = matrix.drop(columns=treated_unit).values
    control_units = matrix.drop(columns=treated_unit).columns.tolist()
    all_dates = prepared.query("is_treated")[date_column].reset_index(drop=True)

    return {
        "prepared_data": prepared,
        "processed_data": matrix,
        "X": X,
        "y": y,
        "control_units": control_units,
        "all_dates": all_dates,
    }

# ====================== Confrormal ======================


def build_conformal_frame(
    panel_data: pd.DataFrame,
    unit_column: str,
    date_column: str,
    outcome_column: str,
    treated_unit: Union[str, int],
    treatment_date: DateLike,
) -> ConformalFrame:
    """Prepare pre/post blocks for conformal inference.

    Args:
        panel_data (pd.DataFrame): Panel dataframe.
        unit_column (str): Unit identifier column.
        date_column (str): Date column.
        outcome_column (str): Outcome variable.
        treated_unit (Union[str, int]): Treated unit identifier.
        treatment_date (DateLike): Treatment date.

    Returns:
        ConformalFrame: Structured pre/post treated/control blocks.
    """
    _ensure_columns(panel_data, [unit_column, date_column, outcome_column])

    treated_unit = str(treated_unit)
    treatment_date = pd.to_datetime(treatment_date)

    prepared = panel_data.copy()
    prepared[unit_column] = prepared[unit_column].astype(str)
    prepared[date_column] = pd.to_datetime(prepared[date_column])

    if treated_unit not in prepared[unit_column].unique():
        raise ValueError(f"Treated unit '{treated_unit}' is not present in panel_data.")

    treated = prepared[prepared[unit_column] == treated_unit].sort_values(date_column)
    controls = prepared[prepared[unit_column] != treated_unit].sort_values(date_column)

    controls_wide = controls.pivot(
        index=date_column, columns=unit_column, values=outcome_column
    ).sort_index()
    pre_mask = controls_wide.index < treatment_date
    post_mask = controls_wide.index >= treatment_date

    return ConformalFrame(
        pre_treatment_treated=treated.loc[
            treated[date_column] < treatment_date, outcome_column
        ].to_numpy(),
        post_treatment_treated=treated.loc[
            treated[date_column] >= treatment_date, outcome_column
        ].to_numpy(),
        pre_treatment_controls=controls_wide.loc[pre_mask],
        post_treatment_controls=controls_wide.loc[post_mask],
        panel=prepared,
    )

# ====================== RDD ======================

def prepare_rdd_data(
    data: pd.DataFrame,
    outcome_column: str,
    running_column: str,
    cutoff: float,
    covariates: List[str] = None,
    bandwidth: float = None,
) -> pd.DataFrame:
    """Prepare dataset for sharp RDD analysis.

    Args:
        data (pd.DataFrame): Source dataframe.
        outcome_column (str): Outcome variable.
        running_column (str): Running/forcing variable.
        cutoff (float): Treatment cutoff on running variable.
        covariates (List[str], optional): Additional covariates.
        bandwidth (float, optional): Optional symmetric bandwidth around cutoff.

    Returns:
        pd.DataFrame: Clean dataframe with centered running variable and treatment flag.
    """
    covariates = covariates or []
    required_columns = [outcome_column, running_column] + covariates
    _ensure_columns(data, required_columns)

    prepared = data[required_columns].copy()
    prepared = prepared.dropna(subset=[outcome_column, running_column] + covariates)

    if not pd.api.types.is_numeric_dtype(prepared[running_column]):
        raise ValueError(f"Column '{running_column}' should be numeric for RDD.")
    if not pd.api.types.is_numeric_dtype(prepared[outcome_column]):
        raise ValueError(f"Column '{outcome_column}' should be numeric for RDD.")

    prepared["running_centered"] = prepared[running_column] - cutoff
    prepared["treatment"] = (prepared[running_column] >= cutoff).astype(int)

    if bandwidth is not None:
        if bandwidth <= 0:
            raise ValueError(f"bandwidth should be positive; got {bandwidth}.")
        prepared = prepared[prepared["running_centered"].abs() <= bandwidth].copy()

    return prepared.reset_index(drop=True)
