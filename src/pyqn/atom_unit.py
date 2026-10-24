# atom_unit.py
# A class for, and attributes used in the parsing of a single "atom unit"
# identified by a base unit, SI-prefix and exponent, such as 'kg', 's-1',
# 'nm3', 'GHz'.
#
# Copyright (C) 2012 Christian Hill
# Department of Physics and Astronomy, University College London
# christian.hill@ucl.ac.uk
#
# This file is part of PyQn
#
# PyQn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyQn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyQn.  If not, see <http://www.gnu.org/licenses/>

import re
from fractions import Fraction
from .si import si_prefixes
from .base_unit import base_unit_stems

# A unit atom is a stem, perhaps with an SI prefix, followed by an optional
# integer or rational exponent, e.g. 'kg', 's-1', 'Pa-1/2'. Exponents can't
# start with 0, so stems ending in 0, such as 'a0' (bohr), are unambiguous.
uatom_patt = re.compile(r"(?P<stem>.+?)(?P<exponent>-?[1-9][0-9]*(?:/[1-9][0-9]*)?)?")


class UnitsError(Exception):
    """
    An Exception class for errors that might occur whilst manipulating units.

    """

    def __init__(self, error_str):
        self.error_str = error_str

    def __str__(self):
        return self.error_str


class AtomUnit(object):
    """
    AtomUnit is a class to represent a single BaseUnit, possibly with an SI
    prefix and possibly raised to some power.

    """

    def __init__(self, prefix, base_unit, exponent=1):
        """Initialize the AtomUnit object."""

        self.base_unit = base_unit
        self.exponent = exponent

        # get the SI prefix (if present)
        self.prefix = prefix
        if prefix is not None:
            try:
                self.si_prefix = si_prefixes[prefix]
            except KeyError:
                raise UnitsError("Invalid or unsupported SI prefix: %s" % prefix)

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
    def parse(self, s_unit_atom):
        """
        Parse the string s_unit_atom into an AtomUnit object and return
        it.

        """

        m = uatom_patt.fullmatch(s_unit_atom)
        if not m:
            raise UnitsError("Invalid unit atom syntax: %s" % s_unit_atom)
        stem, s_exponent = m.group("stem", "exponent")

        # look the stem up as a base unit first (e.g. 'mmHg', 'min'), and only
        # then as a (single-character) SI prefix and a base unit (e.g. 'mm'):
        prefix = None
        if stem not in base_unit_stems:
            prefix, stem = stem[0], stem[1:]
            if prefix not in si_prefixes or stem not in base_unit_stems:
                raise UnitsError("Unrecognised unit: %s" % s_unit_atom)
        base_unit = base_unit_stems[stem]

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
