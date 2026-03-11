import numpy as np
import pandas as pd
import pytest

from cib_advanced_experiments.experiment import Experiment, ExperimentResult


def _build_experiment_data(seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    units = [f"u{i}" for i in range(12)]
    rows = []
    for idx, unit in enumerate(units):
        base = 10 + idx * 0.2
        for date in dates:
            rows.append(
                {
                    "unit": unit,
                    "date": date,
                    "y": base + rng.normal(0.0, 0.1),
                    "retprice": 30 + idx,
                    "age15to24": 0.15 + idx * 0.001,
                }
            )
    return pd.DataFrame(rows)


def _build_treatments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit": ["u0", "u1"],
            "treatment_date": [pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-06")],
        }
    )


def test_experiment_run_and_metrics_success():
    data = _build_experiment_data()
    treatments = _build_treatments()
    control_group = [f"u{i}" for i in range(2, 12)]

    experiment = Experiment(
        data=data,
        treatments_dataframe=treatments,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treatment_date_column="treatment_date",
        date_frequency="D",
        aggregation_statistic="mean",
    )
    result = experiment.run_experiment(
        feature="y",
        to_plot=False,
        control_group=control_group,
        epochs=5,
        lr=0.01,
        inference_method="independent_ttest",
    )

    assert isinstance(result, ExperimentResult)
    metrics = experiment.get_metrics()
    assert "mean_synthetic" in metrics
    assert "mean_real" in metrics
    assert "expected_effect_in_percent" in metrics
    assert "ttest_statistic" in metrics
    assert "ttest_pvalue" in metrics
    assert "test_name" in metrics


def test_experiment_fit_all_and_evaluate_split_api():
    data = _build_experiment_data(seed=102)
    treatments = _build_treatments()
    control_group = [f"u{i}" for i in range(2, 12)]

    experiment = Experiment(
        data=data,
        treatments_dataframe=treatments,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treatment_date_column="treatment_date",
    )
    fitted = experiment.fit_all(
        feature="y",
        to_plot=False,
        control_group=control_group,
        epochs=5,
        lr=0.01,
    )
    assert len(fitted) == 2

    result = experiment.evaluate(
        fitted_units=fitted,
        feature="y",
        quantile_error_border=1.0,
        min_units_after_filter=2,
        min_units_for_inference=2,
        inference_method="paired_ttest",
        to_plot=False,
    )
    assert isinstance(result, ExperimentResult)
    assert result.test_name == "paired_ttest"


def test_experiment_get_aggregated_series_after_fit_all():
    data = _build_experiment_data(seed=107)
    treatments = _build_treatments()
    control_group = [f"u{i}" for i in range(2, 12)]

    experiment = Experiment(
        data=data,
        treatments_dataframe=treatments,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treatment_date_column="treatment_date",
    )
    fitted = experiment.fit_all(
        feature="y",
        to_plot=False,
        control_group=control_group,
        epochs=5,
        lr=0.01,
    )
    aggregated = experiment.get_aggregated_series(feature="y", fitted_units=fitted)

    assert {"date", "real", "synthetic", "effect"}.issubset(aggregated.columns)
    assert len(aggregated) > 0

    expected_last_real = np.mean(
        [
            f.synthetic_control.get_data_on_features("y").iloc[-1]["real"]
            for f in fitted
        ]
    )
    assert np.isclose(aggregated.iloc[-1]["real"], expected_last_real)


def test_experiment_supports_permutation_inference():
    data = _build_experiment_data(seed=103)
    treatments = _build_treatments()
    control_group = [f"u{i}" for i in range(2, 12)]

    experiment = Experiment(
        data=data,
        treatments_dataframe=treatments,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treatment_date_column="treatment_date",
    )
    result = experiment.run_experiment(
        feature="y",
        to_plot=False,
        control_group=control_group,
        epochs=5,
        lr=0.01,
        inference_method="permutation",
        permutation_iterations=200,
        random_state=7,
    )
    assert result.test_name == "permutation"
    assert 0.0 <= result.test_pvalue <= 1.0


def test_experiment_feature_and_propensity_matching_paths():
    data = _build_experiment_data(seed=104)
    treatments = _build_treatments()

    experiment = Experiment(
        data=data,
        treatments_dataframe=treatments,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treatment_date_column="treatment_date",
    )
    feature_res = experiment.run_experiment(
        feature="y",
        to_plot=False,
        number_of_units_for_matching=10,
        matching_method="feature",
        numeric_matching=["retprice", "age15to24"],
        epochs=5,
        lr=0.01,
    )
    assert isinstance(feature_res, ExperimentResult)

    propensity_res = experiment.run_experiment(
        feature="y",
        to_plot=False,
        number_of_units_for_matching=10,
        matching_method="propensity",
        propensity_covariates=["retprice", "age15to24"],
        epochs=5,
        lr=0.01,
    )
    assert isinstance(propensity_res, ExperimentResult)


def test_experiment_raises_on_duplicate_treated_units():
    data = _build_experiment_data()
    treatments = pd.DataFrame(
        {
            "unit": ["u0", "u0"],
            "treatment_date": [pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-07")],
        }
    )

    with pytest.raises(ValueError, match="Each treated unit should appear once"):
        Experiment(
            data=data,
            treatments_dataframe=treatments,
            unit_column="unit",
            date_column="date",
            features=["y"],
            treatment_date_column="treatment_date",
        )


def test_experiment_raises_when_no_synthetic_controls_fitted():
    data = _build_experiment_data()
    treatments = pd.DataFrame(
        {"unit": ["missing_unit"], "treatment_date": [pd.Timestamp("2024-01-06")]}
    )

    experiment = Experiment(
        data=data,
        treatments_dataframe=treatments,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treatment_date_column="treatment_date",
    )
    with pytest.raises(ValueError, match="No synthetic controls were fitted"):
        experiment.run_experiment(
            feature="y",
            to_plot=False,
            control_group=[f"u{i}" for i in range(1, 11)],
            epochs=5,
            lr=0.01,
        )


def test_experiment_quantile_fallback_keeps_min_units():
    data = _build_experiment_data(seed=105)
    treatments = _build_treatments()
    control_group = [f"u{i}" for i in range(2, 12)]

    experiment = Experiment(
        data=data,
        treatments_dataframe=treatments,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treatment_date_column="treatment_date",
    )
    result = experiment.run_experiment(
        feature="y",
        to_plot=False,
        quantile_error_border=0.0,
        min_units_after_filter=1,
        min_units_for_inference=1,
        control_group=control_group,
        epochs=5,
        lr=0.01,
    )
    assert len(result.filtered_frame) == 1


def test_experiment_raises_when_min_units_for_inference_not_met():
    data = _build_experiment_data(seed=106)
    treatments = _build_treatments()
    control_group = [f"u{i}" for i in range(2, 12)]

    experiment = Experiment(
        data=data,
        treatments_dataframe=treatments,
        unit_column="unit",
        date_column="date",
        features=["y"],
        treatment_date_column="treatment_date",
    )
    with pytest.raises(ValueError, match="Need at least 2 units for inference"):
        experiment.run_experiment(
            feature="y",
            to_plot=False,
            quantile_error_border=0.0,
            min_units_after_filter=1,
            min_units_for_inference=2,
            control_group=control_group,
            epochs=5,
            lr=0.01,
        )
