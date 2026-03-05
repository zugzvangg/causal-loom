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
    """Prepared panel blocks for conformal inference on Synthetic Control."""

    pre_treatment_treated: np.ndarray
    post_treatment_treated: np.ndarray
    pre_treatment_controls: pd.DataFrame
    post_treatment_controls: pd.DataFrame
    panel: pd.DataFrame


def _ensure_columns(data: pd.DataFrame, columns: List[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_frequency(date_frequency: DateFrequency) -> None:
    possible_frequencies = {"D", "W", "M", "Y"}
    if date_frequency not in possible_frequencies:
        raise ValueError(
            f"Select date_frequency from {sorted(possible_frequencies)}; got '{date_frequency}'."
        )


def _validate_aggregation(aggregation_statistic: AggregationStatistic) -> None:
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
    """Build a balanced panel for methods that need a row for each unit and each date."""
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
    """Create X/y matrices and helper blocks used by Synthetic Control model fitting."""
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


def build_conformal_frame(
    panel_data: pd.DataFrame,
    unit_column: str,
    date_column: str,
    outcome_column: str,
    treated_unit: Union[str, int],
    treatment_date: DateLike,
) -> ConformalFrame:
    """Prepare pre/post blocks for conformal inference on top of Synthetic Control."""
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


def prepare_rdd_data(
    data: pd.DataFrame,
    outcome_column: str,
    running_column: str,
    cutoff: float,
    covariates: List[str] = None,
    bandwidth: float = None,
) -> pd.DataFrame:
    """Prepare dataset for sharp RDD (center running variable and derive treatment flag)."""
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
