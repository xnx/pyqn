# domains.py
# Domains: named sets of units, which decide which unit strings are valid and
# what they mean, such as the SI units and those accepted for use with them.
#
# Copyright (C) 2026 Christian Hill
# SPDX-License-Identifier: GPL-3.0-or-later

import re
import sys
import warnings
from dataclasses import dataclass, field
from types import MappingProxyType

from .base_unit import base_unit_stems, base_units, metric, v2_stems
from .errors import UnitsError
from .si import si_prefixes


def caller_stacklevel():
    """
    Return the stacklevel, for warnings.warn() in the function calling this
    one, of the first caller outside pyqn, so that warnings point there.

    """

    frame, stacklevel = sys._getframe(1), 1
    while frame is not None and frame.f_globals.get("__name__", "").startswith("pyqn"):
        frame, stacklevel = frame.f_back, stacklevel + 1
    return stacklevel


# A stem can't be empty, contain spaces or the '.' and '/' that join unit
# atoms, or end in a digit 1-9 or '-', which would be read as an exponent.
stem_patt = re.compile(r"1|[^\s./]*[^\s./1-9-]")


@dataclass(frozen=True, eq=False)
class Domain:
    """
    A named set of units, which decides which unit strings are valid and what
    they mean. Each unit is written with its stem, its canonical symbol,
    which means the same in every domain and is what pyqn prints. A domain
    can also accept its field's own spellings of some units, such as 'D' for
    the darcy in geology; format() writes units with these spellings. When a
    domain is created, it checks that every string it accepts reads in only
    one way.

    """

    name: str
    # the BaseUnits in this domain
    units: tuple
    # spellings of units in this domain other than their stems: the first
    # spelling of a unit is the one that format() uses
    spellings: dict = field(default_factory=dict)
    # whether units that existed in pyqn 2.0 still take, with a
    # DeprecationWarning, the SI prefixes that their rules no longer allow:
    # true only for the default domain, everything
    v2_prefixes: bool = False
    # the BaseUnit for each stem or spelling, made from units and spellings
    stems: MappingProxyType = field(init=False, repr=False)

    def __post_init__(self):
        units = tuple(self.units) + tuple(self.spellings.values())
        object.__setattr__(self, "units", tuple(dict.fromkeys(units)))
        object.__setattr__(self, "spellings", MappingProxyType(dict(self.spellings)))

        # the BaseUnit for each stem or spelling
        stems = {}
        for stem, base_unit in [(u.stem, u) for u in self.units] + list(
            self.spellings.items()
        ):
            if not stem_patt.fullmatch(stem):
                raise ValueError(
                    f"Invalid unit symbol in the {self.name} domain: {stem}"
                )
            if stems.setdefault(stem, base_unit) != base_unit:
                raise ValueError(
                    f"{stem} is ambiguous in the {self.name} domain: it is"
                    f" both {stems[stem].name} and {base_unit.name}"
                )
        object.__setattr__(self, "stems", MappingProxyType(stems))

        # check that every string this domain accepts reads in only one way
        spelt = {}
        for stem, base_unit in stems.items():
            for prefix in [None, *self.prefixes(stem)]:
                s_unit = (prefix or "") + stem
                reading = (prefix, base_unit)
                if spelt.setdefault(s_unit, reading) != reading:
                    raise ValueError(
                        f"{s_unit} is ambiguous in the {self.name} domain: it"
                        f" reads as {self.describe(*spelt[s_unit])} and as"
                        f" {self.describe(*reading)}"
                    )

    def prefixes(self, stem):
        """
        Return the SI prefixes that the unit spelt stem takes in this domain,
        including any that are deprecated.

        """

        base_unit = self.stems[stem]
        if base_unit.stem == "1":
            # the unity 'unit' takes no prefix
            return frozenset()
        if self.v2_prefixes and stem in v2_stems:
            # deprecated prefixes give way to units added since pyqn 2.0: ppt
            # is parts per trillion, not the pico-pint
            deprecated = {p for p in metric if p + stem not in self.stems}
            return base_unit.prefixes | deprecated
        return base_unit.prefixes

    @staticmethod
    def describe(prefix, base_unit):
        if prefix is None:
            return base_unit.name
        return f"{si_prefixes[prefix].name}-{base_unit.name}"

    def lookup(self, s_unit):
        """
        Return the SI prefix (or None) and the BaseUnit that the string s_unit
        spells in this domain, such as ('m', <metre>) for 'mm'.

        """

        # look s_unit up as a stem first (e.g. 'mmHg', 'min'), and only then
        # as a (single-character) SI prefix and a stem (e.g. 'mm'):
        if s_unit in self.stems:
            return None, self.stems[s_unit]
        prefix, stem = s_unit[0], s_unit[1:]
        where = "" if self is everything else f" in the {self.name} domain"
        if prefix not in si_prefixes or stem not in self.stems:
            raise UnitsError(f"Unrecognised unit{where}: {s_unit}")
        base_unit = self.stems[stem]
        if prefix not in base_unit.prefixes:
            if prefix not in self.prefixes(stem):
                raise UnitsError(
                    f"Unrecognised unit{where}: {s_unit} (the {base_unit.name},"
                    f" {stem}, does not take the prefix {prefix})"
                )
            warnings.warn(
                f"{s_unit}: the unit {stem} will not take the prefix {prefix}"
                " in pyqn 3.0",
                DeprecationWarning,
                stacklevel=caller_stacklevel(),
            )
        return prefix, base_unit

    def format(self, units):
        """
        Return a string for units (a Units object) using this domain's
        spellings, which parses back to the same units in this domain.

        """

        if units.undef:
            return "undef"
        spelling = {}
        for stem, base_unit in self.spellings.items():
            spelling.setdefault(base_unit, stem)
        s_atoms = []
        for atom_unit in units.atom_units:
            base_unit = atom_unit.base_unit
            if base_unit not in self.units:
                raise UnitsError(
                    f"{base_unit.name} is not a unit in the {self.name} domain"
                )
            s_atom = (atom_unit.prefix or "") + spelling.get(base_unit, base_unit.stem)
            if atom_unit.exponent != 1:
                s_atom += str(atom_unit.exponent)
            s_atoms.append(s_atom)
        return ".".join(s_atoms)


def units_in(*group_names):
    """Return the BaseUnits in the named groups of the table of units."""
    groups = dict(base_units)
    return [base_unit for name in group_names for base_unit in groups[name]]


# The default domain: every unit, plus the standard symbols of units whose
# canonical symbols avoid clashes that pyqn 2.0 had.
everything = Domain(
    "everything",
    units=[base_unit for _, group in base_units for base_unit in group],
    spellings={
        "in": base_unit_stems["inch"],
        "yd": base_unit_stems["yard"],
        "d": base_unit_stems["dy"],
    },
    v2_prefixes=True,
)

# The SI units, and the non-SI units accepted for use with them (SI Brochure,
# 9th edition, Table 8).
si = Domain(
    "si",
    units=units_in("Unity", "SI unit stems", "Derived SI units")
    + [
        base_unit_stems[stem]
        for stem in ("min", "h", "deg", "arcmin", "asec", "ha", "L", "l", "t")
        + ("Da", "u", "eV")
    ],
    spellings={"d": base_unit_stems["dy"], "au": base_unit_stems["AU"]},
)


def field_domain(name, stems, spellings=None, without=()):
    """
    Return a domain of the units in si, except those with the stems in
    without, and the units with the given stems, with si's spellings and any
    others given as a dict of spelling: stem.

    """

    units = [u for u in si.units if u.stem not in without]
    units += [base_unit_stems[stem] for stem in stems]
    spellings = {
        **si.spellings,
        **{s: base_unit_stems[stem] for s, stem in (spellings or {}).items()},
    }
    return Domain(name, units=units, spellings=spellings)


# Field domains: the SI units and those that each field uses with them.
spectroscopy = field_domain(
    "spectroscopy", ("Å", "D", "k", "Torr", "atm", "bar", "erg", "cal", "molec")
)

atomic_and_molecular_physics = field_domain(
    "atomic and molecular physics",
    ("Å", "D", "E_h", "Ry", "a0", "m_e", "e", "hbar", "amu", "b", "molec"),
)

chemistry = field_domain(
    "chemistry",
    ("M", "Å", "D", "cal", "atm", "bar", "Torr", "mmHg", "molec")
    + ("ppm", "ppb", "ppt", "%", "ppmw", "ppbw", "wt%")
    + ("ppmv", "ppbv", "pptv", "vol%", "mol%"),
)

astronomy = field_domain(
    "astronomy",
    ("pc", "ly", "yr", "AU", "Jy", "erg", "dyn", "Å", "G"),
    # the annum, as in Ma and Ga
    spellings={"a": "yr"},
)

geology = field_domain(
    "geology",
    ("yr", "darcy", "Gal", "psi", "bbl", "bar", "atm", "ft", "inch")
    + ("ppm", "ppb", "ppt", "%", "ppmw", "ppbw", "wt%"),
    # the annum, as in Ma, and the darcy, as in mD
    spellings={"a": "yr", "D": "darcy", "in": "inch"},
)

# The units of the Gaussian (cgs) system: their SI equivalents are in the
# default domain, so units in this domain can be converted to them.
gaussian_electromagnetism = Domain(
    "Gaussian electromagnetism",
    units=[
        base_unit_stems[stem]
        for stem in ("1", "m", "g", "s", "dyn", "erg", "G", "Oe", "Mx")
        + ("statC", "statV", "γ")
    ],
    # statC is written as statC, but read as Fr too
    spellings={"statC": base_unit_stems["statC"], "Fr": base_unit_stems["statC"]},
)

radiation_physics = field_domain(
    "radiation physics",
    ("Ci", "R", "rem", "rd"),
    # the rad is the unit of absorbed dose, so this domain has no radian
    spellings={"rad": "rd"},
    without=("rad",),
)

meteorology = field_domain(
    "meteorology",
    ("bar", "atm", "kn", "NM", "ft", "%", "ppm", "ppb", "ppt", "molec")
    + ("ppmv", "ppbv", "pptv", "vol%"),
)

built_in_domains = (
    everything,
    si,
    spectroscopy,
    atomic_and_molecular_physics,
    chemistry,
    astronomy,
    geology,
    gaussian_electromagnetism,
    radiation_physics,
    meteorology,
)
