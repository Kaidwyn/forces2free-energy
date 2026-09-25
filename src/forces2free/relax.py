"""Relax atomic positions and the cell while keeping the space group."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS


@dataclass(frozen=True)
class RelaxResult:
    atoms: Atoms
    converged: bool
    steps: int
    energy_per_atom: float  # eV
    max_force: float  # eV/A
    max_stress: float  # GPa
    spacegroup: str  # after relaxation; checked to equal the starting one


def spacegroup(atoms: Atoms, symprec: float = 1e-5) -> str:
    cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    return f"{dataset.international} ({dataset.number})"


def relax(atoms: Atoms, calc: Calculator, fmax: float = 1e-3, steps: int = 1000) -> RelaxResult:
    """Minimise the energy with respect to positions and cell under FixSymmetry.

    fmax (eV/A) applies to atomic forces and, through FrechetCellFilter, to
    the stress scaled by volume per atom, so 1e-3 eV/A also means a residual
    stress of order 0.01 GPa.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    start = spacegroup(atoms)
    atoms.set_constraint(FixSymmetry(atoms))
    optimizer = BFGS(FrechetCellFilter(atoms), logfile=None)
    converged = bool(optimizer.run(fmax=fmax, steps=steps))
    atoms.set_constraint()

    end = spacegroup(atoms)
    if end != start:
        raise RuntimeError(f"space group changed during relaxation: {start} -> {end}")
    return RelaxResult(
        atoms=atoms,
        converged=converged,
        steps=optimizer.nsteps,
        energy_per_atom=atoms.get_potential_energy() / len(atoms),
        max_force=float(np.abs(atoms.get_forces()).max()),
        max_stress=float(np.abs(atoms.get_stress()).max() / units.GPa),
        spacegroup=end,
    )


def evaluate_fixed(atoms: Atoms, calc: Calculator) -> RelaxResult:
    """Evaluate a structure without relaxing it, e.g. at an experimental lattice constant.

    Meant for structures whose atomic positions are fixed by symmetry (diamond,
    zincblende, rocksalt, fcc), so the forces vanish and only the stress is
    nonzero; it reports the stress the model sees at that lattice constant.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    return RelaxResult(
        atoms=atoms,
        converged=True,
        steps=0,
        energy_per_atom=atoms.get_potential_energy() / len(atoms),
        max_force=float(np.abs(atoms.get_forces()).max()),
        max_stress=float(np.abs(atoms.get_stress()).max() / units.GPa),
        spacegroup=spacegroup(atoms),
    )
