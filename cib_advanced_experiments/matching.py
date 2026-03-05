import datetime
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MatchingResult:
    """Result object for feature-based control selection.

    Attributes:
        control_group (List[str]): Selected control unit identifiers.
        candidate_count (int): Number of candidate controls before final truncation.
        used_exact_matching (bool): Whether categorical exact matching was applied.
        used_numeric_matching (bool): Whether distance-based numeric matching was applied.
    """

    control_group: List[str]
    candidate_count: int
    used_exact_matching: bool
    used_numeric_matching: bool


def _validate_matching_columns(
    data: pd.DataFrame,
    categorical_features: List[str],
    numeric_features: List[str],
) -> None:
    """Validate feature-matching column presence and numeric dtypes.

    Args:
        data (pd.DataFrame): Source dataframe.
        categorical_features (List[str]): Categorical exact-match features.
        numeric_features (List[str]): Numeric distance-match features.

    Returns:
        None
    """
    categorical_features = categorical_features or []
    numeric_features = numeric_features or []

    missing_exact = set(categorical_features) - set(data.columns)
    missing_numeric = set(numeric_features) - set(data.columns)
    if missing_exact:
        raise ValueError(f"Exact matching columns not found: {sorted(missing_exact)}")
    if missing_numeric:
        raise ValueError(
            f"Numeric matching columns not found: {sorted(missing_numeric)}"
        )

    for column in numeric_features:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' is not numeric")


def select_control_group_by_features(
    data: pd.DataFrame,
    unit_column: str,
    date_column: str,
    treated_unit: Union[str, int],
    treatment_date: Union[str, datetime.datetime],
    number_of_units: int,
    categorical_features: List[str] = None,
    numeric_features: List[str] = None,
    fallback_to_all_candidates: bool = True,
) -> MatchingResult:
    """Select control units using feature-based matching.

    Args:
        data (pd.DataFrame): Source dataframe.
        unit_column (str): Unit identifier column.
        date_column (str): Date column.
        treated_unit (Union[str, int]): Treated unit identifier.
        treatment_date (Union[str, datetime.datetime]): Treatment date.
        number_of_units (int): Number of controls to select.
        categorical_features (List[str], optional): Exact-matching categorical features.
        numeric_features (List[str], optional): Numeric features for distance ranking.
        fallback_to_all_candidates (bool, optional): Fallback to all candidates if exact match is empty.

    Returns:
        MatchingResult: Selected controls and matching metadata.
    """
    if number_of_units <= 0:
        raise ValueError(f"number_of_units should be positive; got {number_of_units}.")

    categorical_features = categorical_features or []
    numeric_features = numeric_features or []
    _validate_matching_columns(
        data=data,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )

    treated_unit = str(treated_unit)
    treatment_date = pd.to_datetime(treatment_date)

    prepared = data.copy()
    prepared[unit_column] = prepared[unit_column].astype(str)
    prepared[date_column] = pd.to_datetime(prepared[date_column])

    pre_treatment_data = prepared[prepared[date_column] < treatment_date]
    treated_unit_data = pre_treatment_data[pre_treatment_data[unit_column] == treated_unit]
    if treated_unit_data.empty:
        raise ValueError(f"No pre-treatment data for treated unit {treated_unit}")

    control_candidates = pre_treatment_data[pre_treatment_data[unit_column] != treated_unit]
    used_exact_matching = len(categorical_features) > 0
    if categorical_features:
        treated_values = {
            feature: treated_unit_data.iloc[0][feature] for feature in categorical_features
        }
        exact_mask = control_candidates[categorical_features].apply(
            lambda row: all(row[feature] == treated_values[feature] for feature in categorical_features),
            axis=1,
        )
        control_candidates = control_candidates[exact_mask]

    if control_candidates.empty and fallback_to_all_candidates:
        control_candidates = pre_treatment_data[pre_treatment_data[unit_column] != treated_unit]
    if control_candidates.empty:
        raise ValueError("No control candidates available after matching rules.")

    if numeric_features:
        control_means = (
            control_candidates.groupby(unit_column)[numeric_features].mean().reset_index()
        )
        treated_means = treated_unit_data[numeric_features].mean().values.reshape(1, -1)
        scaler = StandardScaler()
        control_scaled = scaler.fit_transform(control_means[numeric_features].values)
        treated_scaled = scaler.transform(treated_means)
        control_means["distance"] = np.linalg.norm(control_scaled - treated_scaled, axis=1)
        closest = (
            control_means.nsmallest(number_of_units, "distance")[unit_column]
            .astype(str)
            .tolist()
        )
        used_numeric_matching = True
    else:
        closest = control_candidates[unit_column].drop_duplicates().head(number_of_units).astype(str).tolist()
        used_numeric_matching = False

    return MatchingResult(
        control_group=closest,
        candidate_count=control_candidates[unit_column].nunique(),
        used_exact_matching=used_exact_matching,
        used_numeric_matching=used_numeric_matching,
    )
