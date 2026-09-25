"""Workflow for one (model, material) pair, in two steps that share only files.

compute  relax the structure, displace atoms, collect forces
         -> relaxed.vasp, phonopy_params.yaml, compute.json
analyze  rebuild force constants, sample phonons on a mesh, derive the
         thermodynamics and compare it with NIST-JANAF
         -> thermal_properties.csv, comparison_janaf.csv, summary.json

Because the steps talk through files, a model that needs its own Python
environment (dependency conflicts) can run `compute` there while `analyze`
runs here.
"""

from __future__ import annotations

import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase.calculators.calculator import Calculator
from ase.io import write

from forces2free import janaf, phonons
from forces2free.config import Material, Model, results_dir
from forces2free.relax import relax
from forces2free.thermo import harmonic_thermo

MESH = (20, 20, 20)  # Si: S(298 K) within 0.006 J/(K mol) of a 40^3 mesh (scripts/convergence.py)
ROOM_TEMPERATURE = 298.15
TEMPERATURE_STEP = 10.0


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str]:
    return {name: version(name) for name in ("phonopy", "ase", "torch", "mace-torch", "numpy")}


def compute(
    material: Material,
    model: Model,
    calc: Calculator,
    supercell: tuple[int, int, int] | None = None,
    directory: Path | None = None,
    device: str = "",
) -> dict[str, Any]:
    directory = directory or results_dir(model.key, material.key)
    directory.mkdir(parents=True, exist_ok=True)
    supercell = supercell or material.supercell

    start = time.perf_counter()
    relaxed = relax(material.build(), calc)
    relaxed_at = time.perf_counter()
    phonon, residual = phonons.compute_forces(relaxed.atoms, calc, supercell)
    forces_at = time.perf_counter()

    write(directory / "relaxed.vasp", relaxed.atoms, format="vasp", direct=True)
    phonon.save(directory / "phonopy_params.yaml")
    atoms = relaxed.atoms
    info = {
        "model": model.key,
        "material": material.key,
        "relax": {
            "converged": relaxed.converged,
            "steps": relaxed.steps,
            "spacegroup": relaxed.spacegroup,
            "energy_per_atom_eV": relaxed.energy_per_atom,
            "max_force_eV_per_A": relaxed.max_force,
            "max_stress_GPa": relaxed.max_stress,
            "cell_lengths_A": atoms.cell.cellpar()[:3].tolist(),
            "cell_angles_deg": atoms.cell.cellpar()[3:].tolist(),
            "volume_per_atom_A3": atoms.get_volume() / len(atoms),
        },
        "phonons": {
            "supercell": list(supercell),
            "supercell_atoms": len(phonon.supercell),
            "displacements": len(phonon.supercells_with_displacements),
            "displacement_A": phonons.DISPLACEMENT,
            "residual_force_eV_per_A": residual,
        },
        "time_s": {"relax": relaxed_at - start, "forces": forces_at - relaxed_at},
        "device": device,
        "versions": package_versions(),
    }
    _write_json(directory / "compute.json", info)
    return info


def analyze(
    material: Material,
    model_key: str,
    mesh: tuple[int, int, int] = MESH,
    directory: Path | None = None,
) -> dict[str, Any]:
    directory = directory or results_dir(model_key, material.key)
    phonon = phonons.load(directory / "phonopy_params.yaml")
    freqs = phonons.mesh_frequencies(phonon, mesh)
    z = phonons.formula_units(phonon, material.atoms_per_formula)

    temperatures = np.union1d(np.arange(0.0, material.curve_tmax + 1, TEMPERATURE_STEP), [ROOM_TEMPERATURE])
    ours = harmonic_thermo(freqs.frequencies, freqs.weights, temperatures, formula_units=z)
    ref = phonon.run_thermal_properties(temperatures=temperatures, exclude_gamma_acoustic=True)
    table = pd.DataFrame(
        {
            "temperature_K": temperatures,
            "F_kJ_mol": ours.free_energy,
            "U_kJ_mol": ours.internal_energy,
            "S_J_K_mol": ours.entropy,
            "Cv_J_K_mol": ours.heat_capacity,
            "S_phonopy": ref.entropy / z,
            "Cv_phonopy": ref.heat_capacity / z,
        }
    )
    table.to_csv(directory / "thermal_properties.csv", index=False, float_format="%.8g")
    # phonopy uses CODATA 2006 constants, so ~1e-6 differences are expected
    phonopy_diff = max(
        float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
        for a, b in [(ours.entropy[1:], ref.entropy[1:] / z), (ours.heat_capacity[1:], ref.heat_capacity[1:] / z)]
    )

    room = int(np.flatnonzero(temperatures == ROOM_TEMPERATURE)[0])
    summary: dict[str, Any] = {
        "model": model_key,
        "material": material.key,
        "mesh": list(mesh),
        "formula_units_per_primitive_cell": z,
        "max_frequency_THz": float(freqs.frequencies.max()),
        "min_frequency_THz": freqs.min_frequency,
        "imaginary_modes": freqs.n_imaginary,
        "zero_point_energy_kJ_mol": ours.zero_point_energy,
        "S_298_J_K_mol": float(ours.entropy[room]),
        "Cv_298_J_K_mol": float(ours.heat_capacity[room]),
        "phonopy_max_rel_diff": phonopy_diff,
    }
    if material.janaf:
        summary["janaf"] = compare_with_janaf(material, freqs, z, directory)
    _write_json(directory / "summary.json", summary)
    return summary


def compare_with_janaf(
    material: Material, freqs: phonons.MeshFrequencies, z: int, directory: Path
) -> dict[str, Any]:
    """Harmonic Cv and S against experimental Cp and S from 100 K to 0.7 T_melt.

    Cv (constant volume) is not Cp (constant pressure): Cp - Cv = alpha^2 B V T
    is below 1% for Si but ~10% for soft or ionic solids near 800 K. The
    quasi-harmonic Cp is the planned next step.
    """
    exp = janaf.load(material.janaf)
    exp = exp[(exp.temperature >= 100) & (exp.temperature <= material.compare_tmax) & (exp.note == "")]
    calc = harmonic_thermo(freqs.frequencies, freqs.weights, exp.temperature.to_numpy(), formula_units=z)
    table = pd.DataFrame(
        {
            "temperature_K": exp.temperature.to_numpy(),
            "Cv_model": calc.heat_capacity,
            "Cp_exp": exp.Cp.to_numpy(),
            "S_model": calc.entropy,
            "S_exp": exp.S.to_numpy(),
        }
    )
    table["Cp_rel_err_pct"] = 100 * (table.Cv_model / table.Cp_exp - 1)
    table["S_err_J_K_mol"] = table.S_model - table.S_exp
    table.to_csv(directory / "comparison_janaf.csv", index=False, float_format="%.6g")

    room = table[table.temperature_K == ROOM_TEMPERATURE].iloc[0]
    return {
        "table": material.janaf,
        "temperature_range_K": [float(table.temperature_K.min()), float(table.temperature_K.max())],
        "S_298_exp": float(room.S_exp),
        "S_298_err": float(room.S_err_J_K_mol),
        "Cp_298_exp": float(room.Cp_exp),
        "Cv_298_vs_Cp_rel_err_pct": float(room.Cp_rel_err_pct),
        "S_mae_J_K_mol": float(table.S_err_J_K_mol.abs().mean()),
        "Cp_mape_pct": float(table.Cp_rel_err_pct.abs().mean()),
    }
