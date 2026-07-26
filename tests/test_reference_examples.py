import hashlib
from pathlib import Path

import pandas as pd
import pytest

from causal_loom import DifferenceInDifferences

DATA_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "data"
EXPECTED_DATA_HASHES = {
    "billboard_impact.csv": (
        "fc1e2e8ab3e95d6944a96c609b8e373b8569e4677d63bbdc315421702e1a4f3a"
    ),
    "smoking.csv": ("bae31804ce53635822a4bda2b2622d92b707f8c35cf017ea2918a8fdb6083b89"),
}


@pytest.mark.parametrize("filename", sorted(EXPECTED_DATA_HASHES))
def test_bundled_example_data_matches_documented_checksums(filename):
    digest = hashlib.sha256((DATA_DIR / filename).read_bytes()).hexdigest()
    assert digest == EXPECTED_DATA_HASHES[filename]


def test_billboard_example_matches_published_handbook_result():
    """Regression-test the public DiD example against its documented result."""
    data = pd.read_csv(DATA_DIR / "billboard_impact.csv")

    result = DifferenceInDifferences(
        data=data,
        outcome_column="deposits",
        treatment_column="poa",
        post_column="jul",
    ).fit()

    # Matheus Facure, Causal Inference for the Brave and True, chapter 13.
    assert result.estimate == pytest.approx(6.524557692307688, abs=1e-10)
