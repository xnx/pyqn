# atom_unit.py
# A class for, and attributes used in the parsing of a single "atom unit"
# identified by a base unit, SI-prefix and exponent, such as 'kg', 's-1',
# 'nm3', 'GHz'.
#
# Copyright (C) 2012 Christian Hill
# SPDX-License-Identifier: GPL-3.0-or-later

import re
from dataclasses import dataclass
from fractions import Fraction

from . import domains
from .base_unit import BaseUnit
from .errors import UnitsError
from .si import si_prefixes

# A unit atom is a stem, perhaps with an SI prefix, followed by an optional
# integer or rational exponent, e.g. 'kg', 's-1', 'Pa-1/2'. Exponents can't
# start with 0, so stems ending in 0, such as 'a0' (bohr), are unambiguous.
uatom_patt = re.compile(r"(?P<stem>.+?)(?P<exponent>-?[1-9][0-9]*(?:/[1-9][0-9]*)?)?")


@dataclass(frozen=True, slots=True)
class AtomUnit:
    """
    AtomUnit is a class to represent a single BaseUnit, possibly with an SI
    prefix and possibly raised to some power.

    """

    prefix: str | None
    base_unit: BaseUnit
    exponent: int | Fraction = 1

    def __post_init__(self):
        if self.prefix is not None and self.prefix not in si_prefixes:
            raise UnitsError(f"Invalid or unsupported SI prefix: {self.prefix}")

    @property
    def si_prefix(self):
        """The SIPrefix of this AtomUnit, or None if it has no prefix."""
        if self.prefix is None:
            return None
        return si_prefixes[self.prefix]

    @property
    def dims(self):
        """The dimensions of this AtomUnit, as a Dimensions object."""
        return self.base_unit.dims**self.exponent

    @property
    def si_fac(self):
        """The factor relating this AtomUnit to its corresponding SI unit."""
        fac = self.base_unit.fac
        if self.prefix is not None:
            fac *= self.si_prefix.fac
        return fac**self.exponent

    @classmethod
    def parse(self, s_unit_atom, domain=None):
        """
        Parse the string s_unit_atom into an AtomUnit object and return
        it, using the units of domain (by default, domains.everything).

        """

        m = uatom_patt.fullmatch(s_unit_atom)
        if not m:
            raise UnitsError(f"Invalid unit atom syntax: {s_unit_atom}")
        stem, s_exponent = m.group("stem", "exponent")
        prefix, base_unit = (domain or domains.everything).lookup(stem)

        # if there is an exponent, determine what it is (default is 1)
        exponent = 1
        if s_exponent:
            if "/" in s_exponent:
                exponent = Fraction(s_exponent)
            else:
                exponent = int(s_exponent)
        return AtomUnit(prefix, base_unit, exponent)

    def __pow__(self, power):
        """Return the current AtomUnit raised to a specified power."""
        return AtomUnit(self.prefix, self.base_unit, self.exponent * power)

    def __str__(self):
        """String representation of this AtomUnit."""
        s = ""
        if self.prefix:
            s = self.prefix
        s_exponent = ""
        if self.exponent != 1:
            s_exponent = str(self.exponent)
        return "".join([s, str(self.base_unit), s_exponent])

    def __repr__(self):
        return str(self)

    def prefix_base_eq(self, other):
        """
        Compare two AtomUnits and return true if they have the same BaseUnit
        and the same prefix.

        """

        if self.prefix == other.prefix and self.base_unit == other.base_unit:
            return True
        return False
