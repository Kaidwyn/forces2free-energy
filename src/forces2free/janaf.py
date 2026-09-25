"""NIST-JANAF thermochemical tables (NIST Standard Reference Database 13).

M. W. Chase Jr., NIST-JANAF Thermochemical Tables, 4th ed., J. Phys. Chem.
Ref. Data, Monograph 9 (1998). Tables are downloaded on first use from
https://janaf.nist.gov and cached in data/experimental/janaf/.

Columns: temperature (K); Cp, S and -[G - H(298.15 K)]/T in J/(K mol); H - H(298.15 K),
formation enthalpy and formation Gibbs energy in kJ/mol; log10 Kf. Values are
per mole of formula units at a pressure of 1 bar.
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import pandas as pd

from forces2free.config import JANAF_CACHE

URL = "https://janaf.nist.gov/tables/{table}.txt"
COLUMNS = ["temperature", "Cp", "S", "G_H298_T", "H_H298", "dfH", "dfG", "log_Kf"]


def fetch(table: str, cache_dir: Path = JANAF_CACHE) -> Path:
    path = cache_dir / f"{table}.txt"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(URL.format(table=table), headers={"User-Agent": "forces2free"})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def _number(field: str) -> float:
    try:
        return float(field)
    except ValueError:  # "INFINITE" at 0 K, or empty
        return math.nan


def parse(text: str) -> pd.DataFrame:
    """Parse a JANAF text table into one row per temperature.

    A phase change is marked by a row whose last field is a note such as
    "CRYSTAL <--> LIQUID" in place of the formation columns. That row holds
    the values of the lower phase at the transition temperature; `phase`
    counts phases from 0 and increases after it.
    """
    rows = []
    phase = 0
    for line in text.splitlines():
        fields = [f.strip() for f in line.split("\t")]
        temperature = _number(fields[0])
        if math.isnan(temperature):  # title and header lines
            continue
        values = [_number(f) for f in fields[1 : len(COLUMNS)]]
        values += [math.nan] * (len(COLUMNS) - 1 - len(values))
        note = next((f for f in fields[1:] if f and f != "INFINITE" and math.isnan(_number(f))), "")
        rows.append([temperature, *values, phase, note])
        if "<-->" in note:
            phase += 1
    return pd.DataFrame(rows, columns=[*COLUMNS, "phase", "note"])


def load(table: str) -> pd.DataFrame:
    """Rows of the lowest-temperature phase (the crystal for our materials)."""
    data = parse(fetch(table).read_text(encoding="utf-8"))
    return data[data.phase == 0].reset_index(drop=True)
