import numpy as np
import pandas as pd

from cib_advanced_experiments.preprocessing import (
    build_conformal_frame,
    prepare_panel_data,
    prepare_rdd_data,
)
from cib_advanced_experiments.synthetic_control import SyntheticControl


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


def test_prepare_rdd_data_creates_centered_running_and_treatment():
    data = pd.DataFrame(
        {
            "score": [45, 49, 50, 55],
            "conversion": [0.1, 0.2, 0.4, 0.6],
            "segment": [1, 1, 2, 2],
        }
    )
    prepared = prepare_rdd_data(
        data=data,
        outcome_column="conversion",
        running_column="score",
        cutoff=50,
        covariates=["segment"],
        bandwidth=5,
    )

    assert prepared["running_centered"].abs().max() <= 5
    assert prepared.loc[prepared["score"] >= 50, "treatment"].eq(1).all()
    assert prepared.loc[prepared["score"] < 50, "treatment"].eq(0).all()


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
        epochs=5,
        lr=0.01,
    )
    sc.fit()
    weights = sc.get_weights()

    assert weights.shape[0] == 10
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)
