import cib_advanced_experiments


def test_expected_public_api_is_exported():
    expected_exports = {
        "DifferenceInDifferences",
        "Experiment",
        "SynthControlModel",
        "SyntheticControl",
        "SyntheticDifferenceInDifferences",
        "fit_sdid",
        "fit_staggered_sdid",
        "synthetic_difference_in_differences",
    }

    assert expected_exports.issubset(set(cib_advanced_experiments.__all__))
    assert all(hasattr(cib_advanced_experiments, name) for name in expected_exports)
