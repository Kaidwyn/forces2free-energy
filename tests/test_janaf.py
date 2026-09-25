"""The JANAF parser on an excerpt of the silicon table (Si-001), no network needed."""

import math

import pytest

from forces2free.janaf import parse

SI_EXCERPT = """Silicon (Si)\tSi1(ref)
T(K)\tCp\tS\t-[G-H(Tr)]/T\tH-H(Tr)\tdelta-f H\tdelta-f G\tlog Kf
0\t0.\t0.\tINFINITE\t-3.218\t0.\t0.\t0.
100\t7.268\t3.833\t33.351\t-2.952\t0.\t0.\t0.
298.15\t20.000\t18.820\t18.820\t0.\t0.\t0.\t0.
1600\t28.870\t60.261\t39.317\t33.510\t0.\t0.\t0.
1685.000\t29.225\t61.765\t40.412\t35.979\tCRYSTAL <--> LIQUID
1685.000\t27.196\t91.562\t40.412\t86.187\tTRANSITION
1700\t27.196\t91.803\t40.864\t86.595\t0.\t0.\t0.
"""


def test_parse_values_and_units():
    data = parse(SI_EXCERPT)
    room = data[data.temperature == 298.15].iloc[0]
    assert room.Cp == 20.000 and room.S == 18.820 and room.H_H298 == 0.0
    assert math.isnan(data.iloc[0].G_H298_T)  # "INFINITE" at 0 K


def test_phases_split_at_melting():
    data = parse(SI_EXCERPT)
    crystal = data[data.phase == 0]
    assert crystal.temperature.max() == pytest.approx(1685.0)
    assert crystal.iloc[-1].Cp == pytest.approx(29.225)  # crystal side of the melting row
    assert data[data.phase == 1].iloc[0].Cp == pytest.approx(27.196)  # liquid side
    assert list(crystal.note) == ["", "", "", "", "CRYSTAL <--> LIQUID"]
