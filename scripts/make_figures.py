"""Draw the figures in figures/ from the saved results.

    uv run python scripts/make_figures.py
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import constants

from forces2free import phonons
from forces2free.config import FIGURES, load_models, results_dir
from forces2free.plotting import (
    GRID,
    INK_SECONDARY,
    MUTED,
    apply_style,
    experiment_marker,
    header,
    model_colors,
)
from forces2free.thermo import R

# Raman line of Si at room temperature, 520 cm^-1 (the usual spectrometer
# calibration standard), converted with nu = c * wavenumber.
RAMAN_SI_THZ = 520 * constants.c * 100 / 1e12


def _plain(label: str) -> str:
    """phonopy/seekpath mathtext labels ('$\\Gamma$', '$\\mathrm{X}$') -> 'Γ', 'X'."""
    return re.sub(r"\$|\\mathrm\{|\}", "", label).replace("\\Gamma", "Γ")


def _band_ticks(band) -> tuple[list[float], list[str]]:
    labels = [_plain(label) for label in band.labels]
    ticks, names, k = [band.distances[0][0]], [labels[0]], 1
    for i, segment in enumerate(band.distances):
        ticks.append(segment[-1])
        if band.path_connections[i] or i == len(band.distances) - 1:
            names.append(labels[k])
            k += 1
        else:  # jump in the path, e.g. U|K
            names.append(f"{labels[k]}|{labels[k + 1]}")
            k += 2
    return ticks, names


def si_phonons(models, colors) -> None:
    fig, (ax, ax_dos) = plt.subplots(
        1, 2, figsize=(10, 5.2), sharey=True, gridspec_kw={"width_ratios": [3.2, 1], "wspace": 0.04}
    )
    reference_length = None
    gamma_optical = {}
    for model in models:
        phonon = phonons.load(results_dir(model.key, "Si") / "phonopy_params.yaml")
        phonon.auto_band_structure(npoints=101)
        band = phonon.band_structure
        length = band.distances[-1][-1]
        reference_length = reference_length or length
        scale = reference_length / length  # overlay paths of slightly different lattice constants
        for distances, frequencies in zip(band.distances, band.frequencies):
            ax.plot(distances * scale, frequencies, color=colors[model.key], linewidth=1.5)
        if model is models[0]:
            ticks, names = [t * scale for t in _band_ticks(band)[0]], _band_ticks(band)[1]
        phonon.run_qpoints([[0, 0, 0]])
        gamma_optical[model.key] = phonon.qpoints.frequencies[0].max()

        phonon.run_mesh([30, 30, 30], is_gamma_center=True)
        dos = phonon.run_total_dos(freq_pitch=0.04)
        ax_dos.plot(dos.dos, dos.frequency_points, color=colors[model.key], linewidth=1.5)

    gamma_mid = ticks[names.index("Γ", 1)]
    ax.plot([gamma_mid], [RAMAN_SI_THZ], **experiment_marker())
    ax.annotate(
        f"Raman, 300 K\n{RAMAN_SI_THZ:.1f} THz (520 cm⁻¹)",
        (gamma_mid, RAMAN_SI_THZ),
        xytext=(10, 2),
        textcoords="offset points",
        va="center",
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax.set_xticks(ticks, names, fontsize=10.5)
    ax.grid(False, axis="x")
    for t in ticks[1:-1]:
        ax.axvline(t, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(0, 17.5)
    ax.set_ylabel("Frequency (THz)")
    ax_dos.set_xlabel("DOS (states/THz)")
    ax_dos.set_xlim(left=0)
    ax_dos.tick_params(axis="y", labelleft=False)

    handles = [Line2D([], [], color=colors[m.key], label=m.label) for m in models]
    handles.append(Line2D([], [], **experiment_marker(), label="Experiment (Raman)"))
    first = models[0]
    others = [v for k, v in gamma_optical.items() if k != first.key]
    top = header(
        fig,
        "Silicon phonons from four MACE foundation models",
        f"Optical mode at Γ: {first.label} {gamma_optical[first.key]:.1f} THz, the other three "
        f"{min(others):.1f}–{max(others):.1f} THz, experiment {RAMAN_SI_THZ:.1f} THz",
        handles,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.09, top=top)
    fig.savefig(FIGURES / "si_phonons.png", dpi=200)
    plt.close(fig)


def si_thermo(models, colors) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True, gridspec_kw={"hspace": 0.28, "wspace": 0.22})
    (ax_cp, ax_s), (ax_cp_err, ax_s_err) = axes
    for model in models:
        color = colors[model.key]
        curves = pd.read_csv(results_dir(model.key, "Si") / "thermal_properties.csv")
        curves = curves[curves.temperature_K <= 1200]
        ax_cp.plot(curves.temperature_K, curves.Cv_J_K_mol, color=color)
        ax_s.plot(curves.temperature_K, curves.S_J_K_mol, color=color)
        points = pd.read_csv(results_dir(model.key, "Si") / "comparison_janaf.csv")
        points = points[points.temperature_K != 298.15]
        ax_cp_err.plot(points.temperature_K, points.Cp_rel_err_pct, color=color, marker="o", markersize=4.5)
        ax_s_err.plot(points.temperature_K, points.S_err_J_K_mol, color=color, marker="o", markersize=4.5)
    # experiment is the same in every comparison file
    ax_cp.plot(points.temperature_K, points.Cp_exp, **experiment_marker())
    ax_s.plot(points.temperature_K, points.S_exp, **experiment_marker())

    ax_cp.axhline(3 * R, color=MUTED, linewidth=0.9, zorder=1)
    ax_cp.text(20, 3 * R + 0.5, f"3R = {3 * R:.1f}, the harmonic limit", color=INK_SECONDARY, fontsize=9)
    # direct label for the outlier; the three others overlap and rely on the legend
    first = pd.read_csv(results_dir(models[0].key, "Si") / "thermal_properties.csv").set_index("temperature_K")
    ax_s.annotate(models[0].label, (900, first.loc[900.0, "S_J_K_mol"]), xytext=(-8, 4),
                  textcoords="offset points", ha="right", va="bottom", color=INK_SECONDARY, fontsize=9)
    for ax in (ax_cp_err, ax_s_err):
        ax.axhline(0, color=MUTED, linewidth=0.9, zorder=1)

    ax_cp.set_title("Heat capacity: model Cv (harmonic) vs experimental Cp")
    ax_s.set_title("Entropy")
    ax_cp_err.set_title("Cv / Cp − 1")
    ax_s_err.set_title("S(model) − S(experiment)")
    ax_cp.set_ylabel("J K⁻¹ mol⁻¹")
    ax_s.set_ylabel("J K⁻¹ mol⁻¹")
    ax_cp_err.set_ylabel("%")
    ax_s_err.set_ylabel("J K⁻¹ mol⁻¹")
    for ax in axes[1]:
        ax.set_xlabel("Temperature (K)")
    ax_cp.set_xlim(0, 1200)
    ax_cp.set_ylim(0, 29)
    ax_s.set_ylim(bottom=0)

    handles = [Line2D([], [], color=colors[m.key], label=m.label) for m in models]
    handles.append(Line2D([], [], **experiment_marker(), label="NIST-JANAF (1 bar)"))
    top = header(
        fig,
        "Silicon thermodynamics from phonons vs experiment",
        "Harmonic approximation; experiment from NIST-JANAF. Above ~600 K all models fall below "
        "experiment because harmonic Cv cannot exceed 3R.",
        handles,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=top)
    fig.savefig(FIGURES / "si_thermo_vs_janaf.png", dpi=200)
    plt.close(fig)


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    all_models = load_models()
    colors = model_colors(list(all_models))
    models = [m for m in all_models.values() if (results_dir(m.key, "Si") / "summary.json").exists()]
    si_phonons(models, colors)
    si_thermo(models, colors)
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
