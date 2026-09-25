"""Shared figure style.

Colours follow the entity: each model keeps the categorical slot given by its
position in configs/models.yaml in every figure, whatever subset is plotted.
The eight slots are a colour-vision-deficiency-checked palette; experiment is
always drawn in ink so it never competes with a model for a hue.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# diverging scale for signed errors: blue (too low) - grey (zero) - red (too high)
NEGATIVE = "#2a78d6"
NEUTRAL = "#f0efec"
POSITIVE = "#e34948"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 10.5,
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9.5,
        }
    )


def model_colors(model_keys: list[str]) -> dict[str, str]:
    """Slot by configuration order, never by rank or by which models are shown."""
    return {key: SERIES[i] for i, key in enumerate(model_keys)}


def experiment_marker() -> dict:
    """Ink dot with a surface-coloured ring so it stays legible on top of lines."""
    return {"marker": "o", "markersize": 6.5, "markerfacecolor": INK, "markeredgecolor": SURFACE,
            "markeredgewidth": 1.4, "linestyle": "none", "zorder": 5}


def header(fig: plt.Figure, title: str, subtitle: str, handles: list) -> float:
    """Title, subtitle and a one-row legend at fixed distances (in inches) from
    the top, so every figure looks the same whatever its height. Returns the
    figure fraction below which the axes may start."""
    height = fig.get_figheight()

    def below_top(inches: float) -> float:
        return 1 - inches / height

    fig.text(0.01, below_top(0.12), title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, below_top(0.45), subtitle, ha="left", va="top", fontsize=10, color=INK_SECONDARY)
    if not handles:
        return below_top(0.8)
    ncol = len(handles) if len(handles) <= 6 else math.ceil(len(handles) / 2)
    rows = math.ceil(len(handles) / ncol)
    fig.legend(handles=handles, loc="upper left", ncol=ncol, bbox_to_anchor=(0.005, below_top(0.7)),
               handlelength=1.6, columnspacing=1.4)
    return below_top(1.25 + 0.3 * (rows - 1))
