"""Project paths and the material and model registries in configs/."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from ase import Atoms
from ase.build import bulk
from ase.formula import Formula
from ase.spacegroup import crystal

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
JANAF_CACHE = ROOT / "data" / "experimental" / "janaf"


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    formula: str
    structure: dict[str, Any]
    supercell: tuple[int, int, int]
    janaf: str | None
    melting_point: float | None  # K, from the JANAF table; None if it decomposes before melting
    compare_up_to: float | None = None  # K, replaces 0.7 x melting point when set

    @property
    def atoms_per_formula(self) -> int:
        return sum(Formula(self.formula).count().values())

    @property
    def compare_tmax(self) -> float:
        """Highest temperature compared with experiment, 0.7 x melting point by default."""
        if self.compare_up_to is not None:
            return self.compare_up_to
        return 0.7 * self.melting_point

    @property
    def curve_tmax(self) -> float:
        """Upper end of the computed curves: the melting point, rounded up to 100 K."""
        top = self.melting_point or self.compare_tmax / 0.7
        return math.ceil(top / 100) * 100

    def build(self) -> Atoms:
        kind = self.structure["kind"]
        if kind == "bulk":
            return bulk(**self.structure["args"])
        if kind == "crystal":  # from a space group and Wyckoff positions
            return crystal(**self.structure["args"])
        raise ValueError(f"unknown structure kind {kind!r} for {self.key}")


@dataclass(frozen=True)
class Model:
    key: str
    family: str
    checkpoint: str
    label: str
    training_data: str
    functional: str
    license: str


def _read_yaml(name: str) -> dict[str, dict[str, Any]]:
    with open(CONFIGS / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_materials() -> dict[str, Material]:
    materials = {}
    for key, spec in _read_yaml("materials.yaml").items():
        spec = dict(spec, supercell=tuple(spec["supercell"]))
        materials[key] = Material(key=key, **spec)
    return materials


def load_models() -> dict[str, Model]:
    return {key: Model(key=key, **spec) for key, spec in _read_yaml("models.yaml").items()}


def results_dir(model_key: str, material_key: str) -> Path:
    return RESULTS / model_key / material_key
