import datetime
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class PropensityScoreMatchingResult:
    """Result object for propensity-score based control selection.

    Attributes:
        treated_unit (str): Treated unit identifier.
        control_group (List[str]): Selected control unit identifiers.
        scores (pd.DataFrame): Unit-level propensity scores.
    """

    treated_unit: str
    control_group: List[str]
    scores: pd.DataFrame


def _validate_psm_inputs(
    data: pd.DataFrame,
    unit_column: str,
    treatment_column: str,
    covariates: List[str],
) -> None:
    """Validate required columns and binary treatment target.

    Args:
        data (pd.DataFrame): Source dataframe.
        unit_column (str): Unit identifier column.
        treatment_column (str): Binary treatment indicator column.
        covariates (List[str]): Covariates used for propensity model.

    Returns:
        None
    """
    required_columns = [unit_column, treatment_column] + covariates
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[treatment_column].nunique() != 2:
        raise ValueError(f"'{treatment_column}' should be binary (0/1).")


def propensity_score_match(
    data: pd.DataFrame,
    unit_column: str,
    treatment_column: str,
    covariates: List[str],
    treated_unit: str,
    number_of_controls: int,
    caliper: float = None,
    random_state: int = 42,
) -> PropensityScoreMatchingResult:
    """Select nearest controls by propensity score distance.

    Args:
        data (pd.DataFrame): Input dataframe at matching granularity.
        unit_column (str): Unit identifier column.
        treatment_column (str): Binary treatment indicator column.
        covariates (List[str]): Numeric covariates for logistic model.
        treated_unit (str): Treated unit identifier.
        number_of_controls (int): Number of controls to select.
        caliper (float, optional): Max allowed score distance.
        random_state (int, optional): Random seed for logistic model.

    Returns:
        PropensityScoreMatchingResult: Matching result and estimated scores.
    """
    if number_of_controls <= 0:
        raise ValueError(
            f"number_of_controls should be positive; got {number_of_controls}."
        )
    covariates = covariates or []
    if not covariates:
        raise ValueError("Pass at least one covariate for propensity score matching.")
    _validate_psm_inputs(
        data=data,
        unit_column=unit_column,
        treatment_column=treatment_column,
        covariates=covariates,
    )

    prepared = data[[unit_column, treatment_column] + covariates].copy()
    prepared[unit_column] = prepared[unit_column].astype(str)
    prepared = prepared.dropna(subset=[treatment_column] + covariates)
    treated_unit = str(treated_unit)

    if treated_unit not in prepared[unit_column].unique():
        raise ValueError(f"Treated unit '{treated_unit}' is not present in the input.")

    model = LogisticRegression(max_iter=500, random_state=random_state)
    model.fit(prepared[covariates], prepared[treatment_column])
    prepared["propensity_score"] = model.predict_proba(prepared[covariates])[:, 1]

    treated_rows = prepared[prepared[unit_column] == treated_unit]
    if treated_rows.empty:
        raise ValueError(f"No rows available for treated unit '{treated_unit}'.")
    if int(treated_rows[treatment_column].iloc[0]) != 1:
        raise ValueError(
            f"Treated unit '{treated_unit}' should have treatment_column=1 for matching."
        )

    treated_score = float(treated_rows["propensity_score"].mean())
    controls = prepared[prepared[treatment_column] == 0].copy()
    controls["distance"] = (controls["propensity_score"] - treated_score).abs()

    if caliper is not None:
        if caliper <= 0:
            raise ValueError(f"caliper should be positive; got {caliper}.")
        controls = controls[controls["distance"] <= caliper]

    if controls.empty:
        raise ValueError("No controls available after applying caliper filter.")

    control_group = (
        controls.sort_values("distance")[unit_column]
        .drop_duplicates()
        .head(number_of_controls)
        .tolist()
    )
    if len(control_group) < number_of_controls:
        raise ValueError(
            f"Found only {len(control_group)} controls, but requested {number_of_controls}."
        )

    score_frame = prepared[[unit_column, treatment_column, "propensity_score"]].copy()
    return PropensityScoreMatchingResult(
        treated_unit=treated_unit,
        control_group=control_group,
        scores=score_frame,
    )


def select_control_group_by_propensity_from_panel(
    data: pd.DataFrame,
    unit_column: str,
    date_column: str,
    treated_unit: Union[str, int],
    treatment_date: Union[str, datetime.datetime],
    covariates: List[str],
    number_of_controls: int,
    caliper: float = None,
    random_state: int = 42,
) -> PropensityScoreMatchingResult:
    """Run propensity matching on pre-treatment panel aggregates.

    Args:
        data (pd.DataFrame): Panel dataframe.
        unit_column (str): Unit identifier column.
        date_column (str): Date column.
        treated_unit (Union[str, int]): Treated unit identifier.
        treatment_date (Union[str, datetime.datetime]): Treatment date.
        covariates (List[str]): Numeric covariates for propensity model.
        number_of_controls (int): Number of controls to select.
        caliper (float, optional): Max allowed score distance.
        random_state (int, optional): Random seed for logistic model.

    Returns:
        PropensityScoreMatchingResult: Matching result based on pre-period unit summaries.
    """
    covariates = covariates or []
    if not covariates:
        raise ValueError("Pass 'propensity_covariates' for propensity matching.")

    treated_unit = str(treated_unit)
    treatment_date = pd.to_datetime(treatment_date)

    required_columns = [unit_column, date_column] + covariates
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns for propensity matching: {missing}")

    prepared = data[required_columns].copy()
    prepared[unit_column] = prepared[unit_column].astype(str)
    prepared[date_column] = pd.to_datetime(prepared[date_column])

    pre_treatment = prepared[prepared[date_column] < treatment_date].copy()
    if pre_treatment.empty:
        raise ValueError("No pre-treatment data available for propensity matching.")

    for covariate in covariates:
        if not pd.api.types.is_numeric_dtype(pre_treatment[covariate]):
            raise ValueError(f"Propensity covariate '{covariate}' should be numeric.")

    unit_level = pre_treatment.groupby(unit_column, as_index=False)[covariates].mean()
    unit_level["treated"] = (unit_level[unit_column] == treated_unit).astype(int)
    if unit_level["treated"].sum() != 1:
        raise ValueError(
            f"Treated unit '{treated_unit}' is missing in pre-treatment unit-level data."
        )

    return propensity_score_match(
        data=unit_level,
        unit_column=unit_column,
        treatment_column="treated",
        covariates=covariates,
        treated_unit=treated_unit,
        number_of_controls=number_of_controls,
        caliper=caliper,
        random_state=random_state,
    )
