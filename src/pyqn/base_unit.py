# base_unit.py
# A class representing a "base unit", identified as a single unit with no
# prefix or exponent, such as 'g', 'hr', 'bar', 's'.
#
# Copyright (C) 2012-2016 Christian Hill
# SPDX-License-Identifier: GPL-3.0-or-later

import math
from dataclasses import dataclass

from .constants import E_h, a0, c, e, hbar, m_e, m_u
from .dimensions import (
    Dimensions,
    d_area,
    d_charge,
    d_current,
    d_dimensionless,
    d_energy,
    d_force,
    d_length,
    d_magfield_strength,
    d_magnetic_flux,
    d_mass,
    d_pressure,
    d_time,
    d_voltage,
    d_volume,
)
from .si import si_prefixes

# The prefixes that a BaseUnit can take: every SI prefix except hecto (only a
# few units take hecto: 'hbar' would otherwise also be the hectobar), every SI
# prefix, or none. Some units take a few prefixes, listed with the unit.
metric = frozenset(si_prefixes) - {"h"}
metric_and_hecto = frozenset(si_prefixes)
no_prefixes = frozenset()


@dataclass(frozen=True, slots=True)
class BaseUnit:
    """
    A BaseUnit is a commonly-used single unit without a prefix, for example:
    m (metre, length), erg (energy), bar (pressure). BaseUnit objects have
    a description and know their dimensions in terms of powers of Length,
    Time, Mass, etc. (described by a Dimensions object).

    """

    # BaseUnit stem, e.g. 'm', 'g', 'bar', ...
    stem: str
    # BaseUnit name, e.g. 'metre', 'gram', 'bar', ...
    name: str
    # BaseUnit type, e.g. 'length', 'mass', 'pressure', ...
    unit_type: str
    # fac is the relationship between this base unit and the SI base
    # unit, e.g. m: 1. (m is SI), g: 1.e-3 (1 g = 1.e-3 kg),
    # bar: 1.e5 (1 bar = 10e5 Pa)
    fac: float
    # any further description of the BaseUnit
    description: str
    # LaTex representation of the BaseUnit
    latex: str
    # BaseUnit dimensions, as a Dimension object
    dims: Dimensions | None = None
    # the SI prefixes this BaseUnit can take, e.g. 'k', 'M', 'G' for 'yr'
    prefixes: frozenset = no_prefixes

    def __str__(self):
        """String representation of the BaseUnit is just its 'stem'."""
        return self.stem


base_units = (
    (
        "Unity",
        [
            BaseUnit(
                "1",
                "unity",
                "unity",
                1.0,
                "",
                "1",
                d_dimensionless,
                prefixes=no_prefixes,
            ),
        ],
    ),
    (
        "SI unit stems",
        [
            BaseUnit(
                "m",
                "metre",
                "length",
                1.0,
                "",
                "m",
                d_length,
                prefixes=metric_and_hecto,
            ),
            BaseUnit("s", "second", "time", 1.0, "", "s", d_time, prefixes=metric),
            BaseUnit(
                "g", "gram", "mass", 1.0e-3, "", "g", d_mass, prefixes=metric_and_hecto
            ),
            BaseUnit(
                "K",
                "kelvin",
                "temperature",
                1.0,
                "",
                "K",
                Dimensions(Theta=1),
                prefixes=metric,
            ),
            BaseUnit(
                "mol",
                "mole",
                "amount",
                1.0,
                "",
                "mol",
                Dimensions(Q=1),
                prefixes=metric,
            ),
            BaseUnit(
                "A", "amp", "current", 1.0, "", "A", Dimensions(C=1), prefixes=metric
            ),
            BaseUnit(
                "cd",
                "candela",
                "luminous intensity",
                1.0,
                "",
                "cd",
                Dimensions(I=1),
                prefixes=metric,
            ),
        ],
    ),
    (
        "Derived SI units",
        [
            BaseUnit("N", "newton", "force", 1.0, "", "N", d_force, prefixes=metric),
            BaseUnit("J", "joule", "energy", 1.0, "", "J", d_energy, prefixes=metric),
            BaseUnit(
                "W", "watt", "power", 1.0, "", "W", d_energy / d_time, prefixes=metric
            ),
            BaseUnit(
                "Pa",
                "pascal",
                "pressure",
                1.0,
                "",
                "Pa",
                d_pressure,
                prefixes=metric_and_hecto,
            ),
            BaseUnit("C", "coulomb", "charge", 1.0, "", "C", d_charge, prefixes=metric),
            BaseUnit("V", "volt", "voltage", 1.0, "", "V", d_voltage, prefixes=metric),
            BaseUnit(
                "Ω",
                "ohm",
                "electric resistance",
                1,
                "",
                r"\Omega",
                d_voltage / d_current,
                prefixes=metric,
            ),
            BaseUnit(
                "F",
                "farad",
                "capacitance",
                1.0,
                "",
                "F",
                d_charge / d_voltage,
                prefixes=metric,
            ),
            BaseUnit(
                "Wb",
                "weber",
                "magnetic flux",
                1,
                "",
                "Wb",
                d_magnetic_flux,
                prefixes=metric,
            ),
            BaseUnit(
                "H",
                "henry",
                "inductance",
                1,
                "",
                "H",
                d_magnetic_flux / d_current,
                prefixes=metric,
            ),
            BaseUnit(
                "S",
                "siemens",
                "electric conductance",
                1,
                "",
                "S",
                d_current / d_voltage,
                prefixes=metric,
            ),
            BaseUnit(
                "T",
                "tesla",
                "magnetic field strength",
                1.0,
                "",
                "T",
                d_magfield_strength,
                prefixes=metric,
            ),
            BaseUnit(
                "Hz",
                "hertz",
                "cyclic frequency",
                1.0,
                "",
                "Hz",
                d_time**-1,
                prefixes=metric,
            ),
            BaseUnit(
                "rad",
                "radian",
                "angle",
                1.0,
                "",
                "rad",
                d_dimensionless,
                prefixes=metric,
            ),
            BaseUnit(
                "sr",
                "steradian",
                "solid angle",
                1.0,
                "",
                "sr",
                d_dimensionless,
                prefixes=metric,
            ),
            BaseUnit(
                "lm",
                "lumen",
                "luminous flux",
                1.0,
                "",
                "lm",
                Dimensions(I=1),
                prefixes=metric,
            ),
            BaseUnit(
                "lx",
                "lux",
                "illuminance",
                1.0,
                "",
                "lx",
                Dimensions(I=1) / d_area,
                prefixes=metric,
            ),
            BaseUnit(
                "Bq",
                "becquerel",
                "activity",
                1.0,
                "",
                "Bq",
                d_time**-1,
                prefixes=metric,
            ),
            BaseUnit(
                "Gy",
                "gray",
                "absorbed dose",
                1.0,
                "",
                "Gy",
                d_energy / d_mass,
                prefixes=metric,
            ),
            BaseUnit(
                "Sv",
                "sievert",
                "dose equivalent",
                1.0,
                "",
                "Sv",
                d_energy / d_mass,
                prefixes=metric,
            ),
            BaseUnit(
                "kat",
                "katal",
                "catalytic activity",
                1.0,
                "",
                "kat",
                d_time**-1 * Dimensions(Q=1),
                prefixes=metric,
            ),
        ],
    ),
    (
        "Non-SI pressure units",
        [
            BaseUnit(
                "bar", "bar", "pressure", 1.0e5, "", "bar", d_pressure, prefixes=metric
            ),
            BaseUnit(
                "atm",
                "atmosphere",
                "pressure",
                1.01325e5,
                "",
                "atm",
                d_pressure,
                prefixes=no_prefixes,
            ),
            # 1 Torr is exactly 1/760 atm; 1 mmHg is defined as 133.322387415 Pa
            BaseUnit(
                "Torr",
                "torr",
                "pressure",
                101325 / 760,
                "",
                "Torr",
                d_pressure,
                prefixes=metric,
            ),
            BaseUnit(
                "mmHg",
                "millimetres of mercury",
                "pressure",
                133.322387415,
                "",
                "mmHg",
                d_pressure,
                prefixes=no_prefixes,
            ),
        ],
    ),
    (
        "cgs units",
        [
            BaseUnit(
                "dyn", "dyne", "force", 1.0e-5, "", "dyn", d_force, prefixes=metric
            ),
            BaseUnit(
                "erg", "erg", "energy", 1.0e-7, "", "erg", d_energy, prefixes=metric
            ),
            BaseUnit(
                "k",
                "kayser",
                "wavenumber",
                100.0,
                "",
                "k",
                d_length**-1,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "D",
                "debye",
                "electric dipole moment",
                1.0e-21 / c,
                "",
                "D",
                d_charge * d_length,
                prefixes=frozenset("m"),
            ),
            BaseUnit(
                "hbar",
                "hbar",
                "angular momentum",
                hbar,
                "",
                r"\hbar",
                Dimensions(L=2, M=1, T=-1),
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "e",
                "electron charge",
                "charge",
                e,
                "",
                "e",
                d_charge,
                prefixes=no_prefixes,
            ),
        ],
    ),
    (
        "Angular units",
        [
            BaseUnit(
                "deg",
                "degree",
                "angle",
                0.017453292519943295,
                "",
                "deg",
                d_dimensionless,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "arcmin",
                "arcminute",
                "angle",
                2.908882086657216e-4,
                "",
                "arcmin",
                d_dimensionless,
                prefixes=no_prefixes,
            ),
            # NB we can't allow 'as' for arcseconds because of ambiguity with attoseconds
            BaseUnit(
                "asec",
                "arcsecond",
                "angle",
                4.84813681109536e-6,
                "",
                "asec",
                d_dimensionless,
                prefixes=frozenset({"m", "μ"}),
            ),
        ],
    ),
    (
        "Non-SI energy units",
        [
            BaseUnit(
                "eV", "electron volt", "energy", e, "", "eV", d_energy, prefixes=metric
            ),
            BaseUnit(
                "E_h",
                "hartree",
                "energy",
                E_h,
                "",
                "E_h",
                d_energy,
                prefixes=frozenset("m"),
            ),
            BaseUnit(
                "cal",
                "thermochemical calorie",
                "energy",
                4.184,
                "",
                "cal",
                d_energy,
                prefixes=metric,
            ),
            BaseUnit(
                "Ry",
                "rydberg",
                "energy",
                E_h / 2,
                "",
                "Ry",
                d_energy,
                prefixes=frozenset("m"),
            ),
        ],
    ),
    (
        "Non-SI mass units",
        [
            BaseUnit(
                "u",
                "atomic mass unit",
                "mass",
                m_u,
                "",
                "u",
                d_mass,
                prefixes=frozenset("μ"),
            ),
            BaseUnit(
                "amu",
                "atomic mass unit",
                "mass",
                m_u,
                "",
                "amu",
                d_mass,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "Da", "dalton", "mass", m_u, "", "Da", d_mass, prefixes=frozenset("kM")
            ),
            BaseUnit(
                "m_e",
                "electron mass",
                "mass",
                m_e,
                "",
                "m_e",
                d_mass,
                prefixes=no_prefixes,
            ),
        ],
    ),
    (
        "Non-SI units of length, area and volume",
        [
            # Non-SI length units
            BaseUnit(
                "Å",
                "angstrom",
                "length",
                1.0e-10,
                "",
                r"\AA",
                d_length,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "a0", "bohr", "length", a0, "", "a_0", d_length, prefixes=no_prefixes
            ),
            # Non-SI area units
            BaseUnit("b", "barn", "area", 1.0e-28, "", "b", d_area, prefixes=metric),
            # Non-SI volume units
            BaseUnit(
                "l",
                "litre",
                "volume",
                1.0e-3,
                "",
                "l",
                d_volume,
                prefixes=metric_and_hecto,
            ),
            BaseUnit(
                "L",
                "litre",
                "volume",
                1.0e-3,
                "",
                "L",
                d_volume,
                prefixes=metric_and_hecto,
            ),
        ],
    ),
    (
        "Non-SI time units",
        [
            BaseUnit(
                "min", "minute", "time", 60.0, "", "min", d_time, prefixes=no_prefixes
            ),
            BaseUnit(
                "hr", "hour", "time", 3600.0, "", "hr", d_time, prefixes=no_prefixes
            ),
            BaseUnit(
                "h", "hour", "time", 3600.0, "", "h", d_time, prefixes=no_prefixes
            ),
            BaseUnit(
                "dy", "day", "time", 86400.0, "", "d", d_time, prefixes=no_prefixes
            ),
            BaseUnit(
                "yr",
                "year",
                "time",
                31557600.0,
                "",
                "yr",
                d_time,
                prefixes=frozenset("kMG"),
            ),
        ],
    ),
    (
        "Astronomical units",
        [
            BaseUnit(
                "AU",
                "astronomical unit",
                "length",
                1.495978707e11,
                "",
                "AU",
                d_length,
                prefixes=no_prefixes,
            ),
            # 1 pc = 648000/π au
            BaseUnit(
                "pc",
                "parsec",
                "length",
                648000 / math.pi * 1.495978707e11,
                "",
                "pc",
                d_length,
                prefixes=frozenset("kMG"),
            ),
            BaseUnit(
                "ly",
                "light-year",
                "length",
                9.4607304725808e15,
                "",
                "ly",
                d_length,
                prefixes=frozenset("k"),
            ),
        ],
    ),
    (
        "Imperial, customary and US units",
        [
            # NB we can't use 'in' for inch because of a clash with min
            BaseUnit(
                "inch",
                "inch",
                "length",
                0.0254,
                "",
                "in",
                d_length,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "ft", "foot", "length", 0.3048, "", "ft", d_length, prefixes=no_prefixes
            ),
            # NB we can't use 'yd' for yard because of a clash with yd (yoctodays!)
            BaseUnit(
                "yard",
                "yard",
                "length",
                0.9144,
                "",
                "yd",
                d_length,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "fur",
                "furlong",
                "length",
                201.168,
                "",
                "furlong",
                d_length,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "mi",
                "mile",
                "length",
                1609.344,
                "",
                "mi",
                d_length,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "gal",
                "Imperial (UK) gallon",
                "volume",
                4.54609e-3,
                "",
                "gal",
                d_volume,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "pt",
                "Imperial (UK) pint",
                "volume",
                5.6826125e-4,
                "",
                "pt",
                d_volume,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "USgal",
                "US liquid gallon",
                "volume",
                3.785411784e-3,
                "",
                "USgal",
                d_volume,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "USpt",
                "US liquid pint",
                "volume",
                4.73176473e-4,
                "",
                "USpt",
                d_volume,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "st",
                "stone",
                "mass",
                6.35029318,
                "",
                "st",
                d_mass,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "lb",
                "pound",
                "mass",
                0.45359237,
                "",
                "lb",
                d_mass,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "oz",
                "ounce",
                "mass",
                0.028349523125,
                "",
                "oz",
                d_mass,
                prefixes=no_prefixes,
            ),
        ],
    ),
    (
        "Maritime units",
        [
            BaseUnit(
                "NM",
                "nautical mile",
                "length",
                1852.0,
                "",
                "NM",
                d_length,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "kn",
                "knot",
                "speed",
                1852.0 / 3600,
                "",
                "kn",
                d_length / d_time,
                prefixes=no_prefixes,
            ),
        ],
    ),
    (
        "Silly units",
        [
            BaseUnit(
                "fir",
                "firkin",
                "mass",
                40.8233133,
                "",
                "fir",
                d_mass,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "ftn",
                "fortnight",
                "time",
                1.2096e6,
                "",
                "ftn",
                d_time,
                prefixes=no_prefixes,
            ),
        ],
    ),
    (
        "Miscellaneous units",
        [
            BaseUnit(
                "Td",
                "townsend",
                "reduced electric field",
                1.0e-21,
                "",
                "Td",
                d_voltage * d_area,
                prefixes=no_prefixes,
            ),
            BaseUnit(
                "Jy",
                "jansky",
                "spectral flux density",
                1.0e-26,
                "",
                "Jy",
                d_energy / d_area,
                prefixes=metric,
            ),  # W.m-2.Hz-1
        ],
    ),
)

# create a dictionary mapping the BaseUnit stems (as keys) to the BaseUnit
base_unit_stems = {}
for base_unit_group in base_units:
    for base_unit in base_unit_group[1]:
        base_unit_stems[base_unit.stem] = base_unit

# In pyqn 2.0 every unit took every SI prefix except hecto. Until pyqn 3.0 the
# units that existed then still take those prefixes where the rules above no
# longer allow them, but with a DeprecationWarning.
v2_stems = frozenset(base_unit_stems) - {"lm", "lx", "Bq", "Gy", "Sv", "kat"}
