# si.py
# A class representing the SI prefixes (SIPrefix) and a list of the SI
# base units (si_unit_stems): length (L), mass (M), time (T), temperature
# (Theta), amount of substance (Q), current (C) and luminous intensity (I).
#
# Copyright (C) 2012-2016 Christian Hill
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


class SIPrefix(object):
    """A little class describing SI prefixes."""

    def __init__(self, prefix, name, power):
        self.prefix = prefix
        self.name = name
        self.power = power
        self.fac = 10**power


# Here are the SI prefixes that we recognise.
si_prefixes = {
    "y": SIPrefix("y", "yocto", -24),
    "z": SIPrefix("z", "zepto", -21),
    "a": SIPrefix("a", "atto", -18),
    "f": SIPrefix("f", "femto", -15),
    "p": SIPrefix("p", "pico", -12),
    "n": SIPrefix("n", "nano", -9),
    "μ": SIPrefix("μ", "micro", -6),
    "m": SIPrefix("m", "milli", -3),
    "c": SIPrefix("c", "centi", -2),
    "d": SIPrefix("d", "deci", -1),
    "k": SIPrefix("k", "kilo", 3),
    "M": SIPrefix("M", "mega", 6),
    "G": SIPrefix("G", "giga", 9),
    "T": SIPrefix("T", "tera", 12),
    "P": SIPrefix("P", "peta", 15),
    "E": SIPrefix("E", "exa", 18),
    "Z": SIPrefix("Z", "zetta", 21),
    "Y": SIPrefix("Y", "yotta", 24),
}

# The base SI unit stems for length, time, mass, amount of substance,
# thermodynamic temperature, luminous intenstiy and current respectively:
si_unit_stems = ("m", "s", "g", "mol", "K", "cd", "A")
