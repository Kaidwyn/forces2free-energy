"""Every configured starting structure has the intended symmetry and formula units."""

import warnings

import numpy as np
import pytest
from phonopy import Phonopy

from forces2free.config import load_materials
from forces2free.phonons import formula_units, to_phonopy_atoms
from forces2free.relax import spacegroup

# space group, atoms in the built cell, formula units Z in the primitive cell
EXPECTED = {
    "Si": ("Fd-3m (227)", 8, 2),
    "SiC": ("F-43m (216)", 8, 1),
    "AlN": ("P6_3mc (186)", 4, 2),
    "Al": ("Fm-3m (225)", 4, 1),
    "Mg": ("P6_3/mmc (194)", 2, 2),
    "NaCl": ("Fm-3m (225)", 8, 1),
    "NaF": ("Fm-3m (225)", 8, 1),
    "MgO": ("Fm-3m (225)", 8, 1),
    "Al2O3": ("R-3c (167)", 30, 2),
}


def test_every_material_is_checked():
    assert set(load_materials()) == set(EXPECTED)


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_structure(key):
    material = load_materials()[key]
    atoms = material.build()
    group, n_atoms, z = EXPECTED[key]
    assert spacegroup(atoms) == group
    assert len(atoms) == n_atoms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phonon = Phonopy(to_phonopy_atoms(atoms), supercell_matrix=np.eye(3, dtype=int))
    assert formula_units(phonon, material.atoms_per_formula) == z


def test_comparison_range():
    materials = load_materials()
    assert materials["Si"].compare_tmax == pytest.approx(0.7 * 1685)
    assert materials["SiC"].compare_tmax == 1500  # no melting in JANAF, explicit cap
    assert materials["SiC"].curve_tmax >= materials["SiC"].compare_tmax
    assert materials["Al"].curve_tmax == 1000
