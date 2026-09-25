"""The electronic term C_el = S_el = gamma T is added to the phonon values for metals."""

import numpy as np
import pandas as pd
import pytest

from forces2free import janaf, pipeline
from forces2free.config import load_materials
from forces2free.phonons import MeshFrequencies


def test_electronic_term(tmp_path, monkeypatch):
    temperatures = [100.0, 298.15, 600.0]
    experiment = pd.DataFrame({"temperature": temperatures, "Cp": [13.0, 24.2, 27.8],
                               "S": [7.0, 28.3, 47.0], "note": [""] * 3})
    monkeypatch.setattr(janaf, "load", lambda table: experiment)
    freqs = MeshFrequencies(frequencies=np.full((1, 3), 6.0), weights=np.array([1]),
                            min_frequency=0.0, n_imaginary=0)
    al = load_materials()["Al"]
    result = pipeline.compare_with_janaf(al, freqs, 1, tmp_path)
    table = pd.read_csv(tmp_path / "comparison_janaf.csv")
    gamma = 1.35e-3
    np.testing.assert_allclose(table.S_err_J_K_mol_el - table.S_err_J_K_mol, gamma * table.temperature_K, rtol=1e-5)
    assert result["with_electronic"]["S_298_err"] == pytest.approx(result["S_298_err"] + gamma * 298.15, rel=1e-4)

    si = load_materials()["Si"]
    assert "with_electronic" not in pipeline.compare_with_janaf(si, freqs, 2, tmp_path)
