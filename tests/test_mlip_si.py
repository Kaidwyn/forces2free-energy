"""End-to-end run with a real ML potential (MACE-MP-0) on silicon.

Slow and needs the model download on first use, so it only runs on request:

    uv run pytest -m mlip

These tests check that the pipeline is right, not that the model is accurate.
"""

import json

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read
from scipy import constants

from forces2free import phonons, pipeline
from forces2free.config import load_materials, load_models
from forces2free.models import load_calculator
from forces2free.thermo import R

pytestmark = pytest.mark.mlip


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    material = load_materials()["Si"]
    model = load_models()["mace-mp-0"]
    calc = load_calculator(model)
    directory = tmp_path_factory.mktemp("mace-mp-0-Si")
    info = pipeline.compute(material, model, calc, supercell=(2, 2, 2), directory=directory)
    summary = pipeline.analyze(material, model.key, mesh=(12, 12, 12), directory=directory)
    return info, summary, directory, calc


def test_relaxation_keeps_diamond_structure(run):
    info, _, _, _ = run
    relaxed = info["relax"]
    assert relaxed["converged"]
    assert relaxed["spacegroup"] == "Fd-3m (227)"
    assert 5.38 < relaxed["cell_lengths_A"][0] < 5.52  # GGA-trained models, experiment 5.431
    assert info["phonons"]["residual_force_eV_per_A"] < 1e-3


def test_spectrum_is_stable_and_thermodynamics_match_phonopy(run):
    _, summary, _, _ = run
    assert summary["imaginary_modes"] == 0
    assert summary["phonopy_max_rel_diff"] < 2e-5
    assert summary["formula_units_per_primitive_cell"] == 2


def test_units_are_per_mole_of_si(run):
    """Guards against the factor 2 of the two-atom primitive cell, not model quality."""
    _, summary, directory, _ = run
    assert 15 < summary["S_298_J_K_mol"] < 30  # experiment 18.82
    table = np.genfromtxt(directory / "thermal_properties.csv", delimiter=",", names=True)
    assert 0.99 * 3 * R < table["Cv_J_K_mol"][-1] < 3 * R


def test_gamma_optical_mode_agrees_with_frozen_phonon(run):
    """Move the two Si atoms by +d and -d and fit E(d) = E0 + (2 pi nu)^2 m d^2.

    This uses only energies, no phonopy, so it checks the whole
    force-constant route independently.
    """
    _, _, directory, calc = run
    a = read(directory / "relaxed.vasp").cell.cellpar()[0]
    primitive = bulk("Si", "diamond", a=a)
    mass = primitive.get_masses()[0]
    shifts = np.linspace(-0.02, 0.02, 9)
    energies = []
    for d in shifts:
        atoms = primitive.copy()
        atoms.positions[0, 0] += d
        atoms.positions[1, 0] -= d
        atoms.calc = calc
        energies.append(atoms.get_potential_energy())
    curvature = np.polyfit(shifts, energies, 2)[0]  # eV/A^2
    omega = np.sqrt(curvature / mass * constants.e / (constants.atomic_mass * 1e-20))
    frozen = omega / (2 * np.pi) / 1e12  # THz

    phonon = phonons.load(directory / "phonopy_params.yaml")
    phonon.run_qpoints([[0, 0, 0]])
    optical = phonon.qpoints.frequencies[0].max()
    assert frozen == pytest.approx(optical, rel=5e-3)
    assert json.loads((directory / "summary.json").read_text())["max_frequency_THz"] >= optical - 1e-6
