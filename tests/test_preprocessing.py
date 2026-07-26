import numpy as np
import pandas as pd
import pytest

from causal_loom.preprocessing import (
    build_conformal_frame,
    build_synthetic_control_matrices,
    prepare_panel_data,
)
from causal_loom.synthetic_control import SyntheticControl


def test_prepare_panel_data_builds_balanced_panel():
    data = pd.DataFrame(
        {
            "unit": ["A", "A", "B"],
            "date": ["2024-01-01", "2024-01-03", "2024-01-01"],
            "y": [1.0, 3.0, 2.0],
        }
    )

    prepared = prepare_panel_data(
        data=data,
        unit_column="unit",
        date_column="date",
        value_columns=["y"],
        min_date="2024-01-01",
        max_date="2024-01-03",
        date_frequency="D",
        aggregation_statistic="mean",
    )

    assert len(prepared) == 6
    assert prepared["unit"].nunique() == 2
    assert prepared["date"].nunique() == 3
    assert (prepared["y"] == 0.0).any()


def test_build_conformal_frame_splits_pre_and_post():
    panel = pd.DataFrame(
        {
            "unit": ["A", "A", "A", "B", "B", "B"],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "y": [1.0, 1.1, 1.2, 0.8, 0.9, 1.0],
        }
    )
    frame = build_conformal_frame(
        panel_data=panel,
        unit_column="unit",
        date_column="date",
        outcome_column="y",
        treated_unit="A",
        treatment_date="2024-01-03",
    )

    assert frame.pre_treatment_treated.shape[0] == 2
    assert frame.post_treatment_treated.shape[0] == 1
    assert frame.pre_treatment_controls.shape[0] == 2
    assert frame.post_treatment_controls.shape[0] == 1


def test_synthetic_control_accepts_unbalanced_source_data():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    units = [f"u{i}" for i in range(12)]
    rows = []
    rng = np.random.default_rng(0)
    for unit in units:
        for date in dates:
            rows.append({"unit": unit, "date": date, "y": float(rng.normal())})

    data = pd.DataFrame(rows)
    # Drop a few rows to emulate sparse event-level source data.
    data = data[~((data["unit"] == "u5") & (data["date"].isin(dates[:2])))]

    sc = SyntheticControl(
        data=data,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treated_unit="u0",
        treatment_date="2024-01-06",
        control_group=[f"u{i}" for i in range(1, 11)],
        min_date="2024-01-01",
        max_date="2024-01-08",
        max_iter=100,
        tolerance=1e-8,
    )
    sc.fit()
    weights = sc.get_weights()

    assert weights.shape[0] == 10
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)


def test_prepare_panel_data_raises_on_missing_columns():
    data = pd.DataFrame({"unit": ["A"], "date": ["2024-01-01"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_panel_data(
            data=data,
            unit_column="unit",
            date_column="date",
            value_columns=["y"],
        )


def test_prepare_panel_data_raises_on_invalid_date_range():
    data = pd.DataFrame(
        {"unit": ["A", "A"], "date": ["2024-01-01", "2024-01-02"], "y": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="min_date should be less than max_date"):
        prepare_panel_data(
            data=data,
            unit_column="unit",
            date_column="date",
            value_columns=["y"],
            min_date="2024-01-03",
            max_date="2024-01-02",
        )


def test_build_synthetic_control_matrices_raises_on_missing_treated_unit():
    panel = pd.DataFrame(
        {
            "unit": ["A", "A", "B", "B"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            "y": [1.0, 2.0, 1.5, 1.6],
        }
    )
    with pytest.raises(ValueError, match="is not present in panel_data"):
        build_synthetic_control_matrices(
            panel_data=panel,
            unit_column="unit",
            date_column="date",
            features=["y"],
            treated_unit="missing",
            treatment_date="2024-01-02",
        )


def test_build_conformal_frame_raises_on_missing_treated_unit():
    panel = pd.DataFrame(
        {
            "unit": ["A", "A", "B", "B"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            "y": [1.0, 2.0, 1.5, 1.6],
        }
    )
    with pytest.raises(ValueError, match="is not present in panel_data"):
        build_conformal_frame(
            panel_data=panel,
            unit_column="unit",
            date_column="date",
            outcome_column="y",
            treated_unit="missing",
            treatment_date="2024-01-02",
        )


def test_prepare_panel_data_accepts_yearly_alias_y():
    data = pd.DataFrame(
        {
            "unit": ["A", "A", "B", "B"],
            "date": pd.to_datetime(
                ["2023-01-01", "2024-01-01", "2023-01-01", "2024-01-01"]
            ),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    prepared = prepare_panel_data(
        data=data,
        unit_column="unit",
        date_column="date",
        value_columns=["y"],
        date_frequency="Y",
        aggregation_statistic="mean",
    )
    assert not prepared.empty
