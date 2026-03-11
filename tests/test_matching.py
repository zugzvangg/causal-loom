import numpy as np
import pandas as pd
import pytest

from cib_advanced_experiments.matching import select_control_group_by_features
from cib_advanced_experiments.propensity_score_matching import (
    propensity_score_match,
    select_control_group_by_propensity_from_panel,
)


def test_select_control_group_by_features_returns_closest_controls():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    rows = []
    for unit_id, base in [("u0", 10.0), ("u1", 10.1), ("u2", 9.8), ("u3", 15.0), ("u4", 10.3)]:
        for date in dates:
            rows.append(
                {
                    "unit": unit_id,
                    "date": date,
                    "revenue": base,
                    "segment": "A",
                }
            )
    data = pd.DataFrame(rows)

    result = select_control_group_by_features(
        data=data,
        unit_column="unit",
        date_column="date",
        treated_unit="u0",
        treatment_date="2024-01-04",
        number_of_units=2,
        categorical_features=["segment"],
        numeric_features=["revenue"],
    )

    assert len(result.control_group) == 2
    assert result.control_group == ["u1", "u2"]
    assert result.used_exact_matching is True
    assert result.used_numeric_matching is True


def test_propensity_score_match_returns_requested_control_group():
    data = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(10)],
            "treated": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "x1": [0.9, 0.88, 0.7, 0.2, 0.1, 0.8, 0.3, 0.4, 0.6, 0.5],
            "x2": [1.0, 0.95, 0.65, 0.1, 0.2, 0.75, 0.25, 0.35, 0.55, 0.45],
        }
    )

    result = propensity_score_match(
        data=data,
        unit_column="unit",
        treatment_column="treated",
        covariates=["x1", "x2"],
        treated_unit="u0",
        number_of_controls=3,
    )

    assert result.treated_unit == "u0"
    assert len(result.control_group) == 3
    assert "propensity_score" in result.scores.columns


def test_select_control_group_by_features_raises_on_non_positive_number_of_units():
    data = pd.DataFrame(
        {
            "unit": ["u0", "u1"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "revenue": [10.0, 10.5],
        }
    )
    with pytest.raises(ValueError, match="number_of_units should be positive"):
        select_control_group_by_features(
            data=data,
            unit_column="unit",
            date_column="date",
            treated_unit="u0",
            treatment_date="2024-01-02",
            number_of_units=0,
            numeric_features=["revenue"],
        )


def test_select_control_group_by_features_raises_on_non_numeric_numeric_feature():
    data = pd.DataFrame(
        {
            "unit": ["u0", "u1", "u0", "u1"],
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
            ),
            "segment": ["A", "A", "A", "A"],
        }
    )
    with pytest.raises(ValueError, match="is not numeric"):
        select_control_group_by_features(
            data=data,
            unit_column="unit",
            date_column="date",
            treated_unit="u0",
            treatment_date="2024-01-02",
            number_of_units=1,
            numeric_features=["segment"],
        )


def test_select_control_group_by_features_raises_when_exact_match_empty_and_no_fallback():
    data = pd.DataFrame(
        {
            "unit": ["u0", "u1", "u2", "u0", "u1", "u2"],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "segment": ["A", "B", "B", "A", "B", "B"],
            "revenue": [10.0, 8.0, 7.5, 10.2, 8.2, 7.2],
        }
    )
    with pytest.raises(ValueError, match="No control candidates available"):
        select_control_group_by_features(
            data=data,
            unit_column="unit",
            date_column="date",
            treated_unit="u0",
            treatment_date="2024-01-02",
            number_of_units=1,
            categorical_features=["segment"],
            fallback_to_all_candidates=False,
        )


def test_propensity_score_match_raises_on_invalid_inputs():
    base = pd.DataFrame(
        {
            "unit": ["u0", "u1", "u2"],
            "treated": [1, 0, 0],
            "x1": [0.9, 0.6, 0.5],
            "x2": [1.0, 0.7, 0.4],
        }
    )
    with pytest.raises(ValueError, match="at least one covariate"):
        propensity_score_match(
            data=base,
            unit_column="unit",
            treatment_column="treated",
            covariates=[],
            treated_unit="u0",
            number_of_controls=1,
        )

    not_treated = base.copy()
    not_treated.loc[0, "treated"] = 0
    not_treated.loc[1, "treated"] = 1
    with pytest.raises(ValueError, match="should have treatment_column=1"):
        propensity_score_match(
            data=not_treated,
            unit_column="unit",
            treatment_column="treated",
            covariates=["x1", "x2"],
            treated_unit="u0",
            number_of_controls=1,
        )


def test_select_control_group_by_propensity_from_panel_raises_on_bad_panel_inputs():
    panel = pd.DataFrame(
        {
            "unit": ["u0", "u1", "u2", "u0", "u1", "u2"],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "x1": [1.0, 0.9, 0.8, 1.1, 0.95, 0.82],
            "segment": ["A", "A", "B", "A", "A", "B"],
        }
    )
    with pytest.raises(ValueError, match="Propensity covariate 'segment' should be numeric"):
        select_control_group_by_propensity_from_panel(
            data=panel,
            unit_column="unit",
            date_column="date",
            treated_unit="u0",
            treatment_date="2024-01-03",
            covariates=["segment"],
            number_of_controls=1,
        )

    with pytest.raises(ValueError, match="No pre-treatment data available"):
        select_control_group_by_propensity_from_panel(
            data=panel,
            unit_column="unit",
            date_column="date",
            treated_unit="u0",
            treatment_date="2023-12-01",
            covariates=["x1"],
            number_of_controls=1,
        )
