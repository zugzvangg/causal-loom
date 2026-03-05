import numpy as np
import pandas as pd

from cib_advanced_experiments.matching import select_control_group_by_features
from cib_advanced_experiments.propensity_score_matching import propensity_score_match


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



