"""Shared helpers for DiD-style estimators."""

import numpy as np
import pandas as pd


def validate_binary_column(data: pd.DataFrame, column: str) -> None:
    """Validate that a column contains only binary values {0, 1}.

    Args:
        data (pd.DataFrame): Input dataframe.
        column (str): Column name to validate.

    Returns:
        None
    """
    values = sorted(data[column].dropna().unique().tolist())
    if set(values) != {0, 1}:
        raise ValueError(
            f"Column '{column}' should be binary with values {{0, 1}}, got {values}."
        )


def safe_mean(data: pd.DataFrame, mask: pd.Series, outcome_column: str) -> float:
    """Compute unweighted mean with empty-slice guard.

    Args:
        data (pd.DataFrame): Source dataframe.
        mask (pd.Series): Boolean mask for subset selection.
        outcome_column (str): Outcome column name.

    Returns:
        float: Mean value for selected rows.
    """
    subset = data.loc[mask, outcome_column]
    if subset.empty:
        raise ValueError(
            "Each DiD cell should have at least one observation. "
            "Check treatment/post assignment."
        )
    return float(subset.mean())


def safe_weighted_mean(
    data: pd.DataFrame,
    mask: pd.Series,
    outcome_column: str,
    weight_column: str,
) -> float:
    """Compute weighted mean with safety checks for empty/zero-weight slices.

    Args:
        data (pd.DataFrame): Source dataframe.
        mask (pd.Series): Boolean mask for subset selection.
        outcome_column (str): Outcome column name.
        weight_column (str): Weight column name.

    Returns:
        float: Weighted mean of outcome for selected rows.
    """
    subset = data.loc[mask, [outcome_column, weight_column]]
    if subset.empty:
        raise ValueError(
            "Each SDID cell should have at least one observation. "
            "Check treatment/post assignment."
        )

    weights = subset[weight_column].astype(float).to_numpy()
    values = subset[outcome_column].astype(float).to_numpy()
    total_weight = float(weights.sum())
    if np.isclose(total_weight, 0.0):
        raise ValueError("Weights sum to zero in one of SDID cells.")
    return float(np.average(values, weights=weights))
