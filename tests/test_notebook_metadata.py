import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebooks_are_portable_and_have_no_saved_errors(notebook_path):
    serialized_notebook = notebook_path.read_text(encoding="utf-8")
    notebook = json.loads(serialized_notebook)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    saved_errors = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    unexecuted_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "".join(cell.get("source", [])).strip()
        and cell.get("execution_count") is None
    ]

    assert "/Users/" not in serialized_notebook
    assert "/home/" not in serialized_notebook
    assert "THIRD_PARTY_NOTICES.md" in source
    assert not saved_errors
    assert not unexecuted_cells
