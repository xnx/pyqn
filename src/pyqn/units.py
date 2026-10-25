# units.py
# A class representing the units of a physical quantity.
#
# Copyright (C) 2012-2016 Christian Hill
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import math
import re
import unicodedata
from fractions import Fraction

from .atom_unit import AtomUnit, UnitsError
from .constants import NA, c, h, kB
from .dimensions import (
    Dimensions,
    d_amount_fraction,
    d_dimensionless,
    d_energy,
    d_length,
    d_mass_fraction,
    d_temperature,
    d_time,
    d_volume_fraction,
)


class UnitsConversionError(UnitsError):
    pass


# Units objects are immutable; __init__, __eq__, __hash__ and __repr__ are
# defined below.
@dataclasses.dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Units:
    """
    A class to represent the units of a physical quantity.

    """

    atom_units: tuple
    undef: bool
    dims: Dimensions | None

    def __init__(self, units, domain=None):
        """
        Initialize this Units from another Units object, a list or tuple of
        AtomUnit objects, or by parsing a string representing the units with
        the units of domain (by default, domains.everything).

        """

        undef = False
        if isinstance(units, Units):
            atom_units, undef = units.atom_units, units.undef
        elif isinstance(units, str):
            if units == "undef":
                atom_units, undef = (), True
            else:
                atom_units = self.parse(units, domain=domain).atom_units
        elif isinstance(units, (list, tuple)):
            atom_units = tuple(units)
        else:
            raise TypeError(
                "Attempt to initialize Units object with"
                f" argument units of type {type(units)}"
            )
        object.__setattr__(self, "atom_units", atom_units)
        object.__setattr__(self, "undef", undef)
        # also get the dimensions of the units
        object.__setattr__(self, "dims", None if undef else self.get_dims())

    def has_units(self):
        if self.undef or self.atom_units:
            return True
        return False

    def get_dims(self):
        """
        Return a Dimensions object representing the dimensions of this
        Units.

        """

        if self.undef:
            raise UnitsError("Undefined units have no dimensions")
        dims = Dimensions()
        for atom_unit in self.atom_units:
            dims *= atom_unit.dims
        return dims

    @classmethod
    def parse(self, s_compoundunit, no_divided_units=False, domain=None):
        """
        Parse the string s_compoundunit and return the corresponding
        Units object, using the units of domain (by default,
        domains.everything).

        """

        # Read look-alike characters such as the micro sign (U+00B5) and the
        # ohm sign (U+2126) as the characters in the unit table (U+03BC, U+03A9).
        s_compoundunit = unicodedata.normalize("NFKC", s_compoundunit)
        if re.search(r"\s", s_compoundunit):
            raise UnitsError(f"Units cannot contain spaces: {s_compoundunit!r}")

        if no_divided_units:
            # s_compoundunit does not consist of any units separated by '/'
            # so parse immediately as a sequence of multiplied units.
            return Units.parse_mult_units(s_compoundunit, domain=domain)

        # We need to temporarily identify rational exponents (e.g. "Pa-1/2")
        # and replace the / with : so that we can split up the atomic units
        # properly.
        patt = r"\d+/\d+"
        rational_exponents = re.findall(patt, s_compoundunit)
        for e in rational_exponents:
            es = e.replace("/", ":")
            s_compoundunit = s_compoundunit.replace(e, es)

        div_fields = s_compoundunit.split("/")
        # don't forget to put back the "/" character in any rational exponents.
        compound_unit = Units.parse_mult_units(
            div_fields[0].replace(":", "/"), domain=domain
        )
        for div_field in div_fields[1:]:
            div_field = div_field.replace(":", "/")
            compound_unit = compound_unit / Units.parse(
                div_field, no_divided_units=True, domain=domain
            )
        return compound_unit

    @classmethod
    def parse_mult_units(self, munit, domain=None):
        """
        Parse a string of units multiplied together (indicated by '.'),
        returning the corresponding Units object.

        """
        atom_units = []
        for s_unit in munit.split("."):
            atom_unit = AtomUnit.parse(s_unit, domain=domain)
            if atom_unit.base_unit.stem != "1":
                # the unity 'unit' is not really a unit
                atom_units.append(atom_unit)
        return Units(atom_units)

    def _times_atoms(self, other_atom_units):
        """
        Return the AtomUnits of this Units multiplied by other_atom_units: the
        exponents of AtomUnits with the same prefix and BaseUnit are added,
        and AtomUnits that cancel are dropped.

        """

        atom_units = list(self.atom_units)
        for other_atom_unit in other_atom_units:
            for i, atom_unit in enumerate(atom_units):
                if atom_unit.prefix_base_eq(other_atom_unit):
                    # this prefix and BaseUnit is already in the AtomUnits:
                    # update its exponent
                    exponent = atom_unit.exponent + other_atom_unit.exponent
                    if exponent == 0:
                        # this AtomUnit has cancelled:
                        del atom_units[i]
                    else:
                        atom_units[i] = dataclasses.replace(
                            atom_unit, exponent=exponent
                        )
                    break
            else:
                # new prefix and BaseUnit; append to the AtomUnits
                atom_units.append(other_atom_unit)
        return atom_units

    @staticmethod
    def _as_units(other):
        """
        Return other as a Units object if it is one, a string or the number 1;
        otherwise return None.

        """

        if isinstance(other, Units):
            return other
        if isinstance(other, str):
            return Units(other)
        if other == 1:
            return Units([])
        return None

    def __mul__(self, other):
        """Return the product of this Units object with another."""

        other = self._as_units(other)
        if other is None:
            return NotImplemented
        if self.undef or other.undef:
            # a product involving undefined units is undefined
            return Units("undef")
        return Units(self._times_atoms(other.atom_units))

    def __rmul__(self, other):
        other = self._as_units(other)
        if other is None:
            return NotImplemented
        return other * self

    def __truediv__(self, other):
        """Return the ratio of this Units divided by another."""

        other = self._as_units(other)
        if other is None:
            return NotImplemented
        if self.undef or other.undef:
            # a ratio involving undefined units is undefined
            return Units("undef")
        return Units(self._times_atoms(atom_unit**-1 for atom_unit in other.atom_units))

    def __rtruediv__(self, other):
        other = self._as_units(other)
        if other is None:
            return NotImplemented
        return other / self

    def __pow__(self, power):
        # any units, even undefined ones, raised to the power 0 are dimensionless
        if power == 0:
            return Units([])
        if self.undef:
            return Units("undef")
        result_atom_units = []
        for atom_unit in self.atom_units:
            result_atom_units.append(atom_unit**power)
        return Units(result_atom_units)

    def __repr__(self):
        """String representation of this Units."""
        if self.undef:
            return "undef"
        return ".".join([str(atom_unit) for atom_unit in self.atom_units])

    __str__ = __repr__

    def __eq__(self, other):
        """Test for equality with another Units object, or with 1."""

        if not isinstance(other, Units):
            if other != 1:
                return NotImplemented
            other = Units([])

        if self.undef or other.undef:
            return False

        if self.get_dims() != other.get_dims():
            # obviously the units aren't the same if they have different dimensions
            return False
        # if two Units objects have the same dimensions, they are equal if
        # their conversion factors to SI units are the same:
        return math.isclose(self.to_si(), other.to_si(), rel_tol=1.0e-9)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        # Units that are equal have the same dimensions, and units equal to 1
        # are dimensionless, so hashing the dimensions is consistent with
        # __eq__. Undefined units are equal to nothing, not even themselves.
        if self.undef:
            return object.__hash__(self)
        if self.dims == d_dimensionless:
            return hash(1)
        return hash(self.dims)

    def to_si(self):
        """Return the factor needed to convert this Units to SI."""
        if self.undef:
            raise UnitsError("Undefined units cannot be converted to SI")
        fac = 1.0
        for atom_unit in self.atom_units:
            fac *= atom_unit.si_fac
        return fac

    @property
    def html(self):
        if self.undef:
            return "undef"

        h = []
        n = len(self.atom_units)
        for i, atom_unit in enumerate(self.atom_units):
            h.extend([atom_unit.prefix or "", atom_unit.base_unit.stem])
            if atom_unit.exponent != 1:
                h.append(f"<sup>{atom_unit.exponent}</sup>")
            if i < n - 1:
                h.append(" ")
        return "".join(h)

    @property
    def latex(self):
        if self.undef:
            return r"\mathrm{undef}"

        e = []
        n = len(self.atom_units)
        for i, atom_unit in enumerate(self.atom_units):
            # TODO use proper LaTeX for prefix.
            e.extend([atom_unit.prefix or "", atom_unit.base_unit.latex])
            if atom_unit.exponent != 1:
                e.append("^{" + str(atom_unit.exponent) + "}")
            if i < n - 1:
                e.append(r"\,")
        return r"\mathrm{" + "".join(e) + "}"

    def conversion(self, other, force=None):
        """
        Return the factor required to convert this Units to another. Their
        dimensions have to match, unless force names one or more conventions,
        each of which treats some physical constants as 1, so that units of
        different dimensions can be interconverted:

        force='spec': h and c, for spectroscopy: energy, frequency and
            wavenumber are interconverted with E = h.nu = hc.nu~ (and so
            are lengths and light-travel times, as c is 1);
        force='kBT': the Boltzmann constant: energy and temperature are
            interconverted with E = kB.T;
        force='mol': the Avogadro constant: molar quantities and those per
            entity are interconverted, as between kJ.mol-1 and J; the amount
            dimension must be added or removed in its entirety;
        force='mass fraction', 'volume fraction' or 'amount fraction': treats
            that pseudo-dimension as dimensionless, as between ppmv and
            umol/mol.

        Conventions can be combined, as in force=('mol', 'spec'): the
        constants have independent dimensions, so there is at most one way
        to use them in a conversion.

        """

        if isinstance(other, str):
            other = Units(other)
        if self.undef or other.undef:
            raise UnitsError(f"Cannot convert between {self} and {other}")

        conventions = (force,) if isinstance(force, str) else tuple(force or ())
        for convention in conventions:
            if convention not in convention_constants:
                raise UnitsError(f"Unknown units conversion convention: {convention}")

        fac = self.to_si() / other.to_si()
        self_dims, other_dims = self.get_dims(), other.get_dims()
        if self_dims == other_dims:
            return fac
        if not conventions:
            raise UnitsError(
                f"Failure in units conversion: units {self} ({self_dims}) and"
                f" {other} ({other_dims}) have different dimensions"
            )

        # find the powers of the constants by which the dimensions differ
        names = list(
            dict.fromkeys(name for c in conventions for name in convention_constants[c])
        )
        powers = powers_of(
            self_dims / other_dims, [physical_constants[name][1] for name in names]
        )
        if powers is None:
            raise UnitsConversionError(
                f"Cannot convert {self} ({self_dims}) to {other} ({other_dims})"
                f" with the conventions {', '.join(conventions)}"
            )
        powers = dict(zip(names, powers))
        if powers.get("NA") and self_dims.dims[4] and other_dims.dims[4]:
            # the amount dimension can only be added or removed in its entirety
            raise UnitsConversionError(
                f"Cannot force molar conversion between {self} and {other}"
            )
        for name, power in powers.items():
            fac /= physical_constants[name][0] ** power
        return fac

    def kBT_conversion(self, other):
        """Return the factor to convert this Units to other with force='kBT'."""
        return self.conversion(other, force="kBT")

    def mol_conversion(self, other):
        """Return the factor to convert this Units to other with force='mol'."""
        return self.conversion(other, force="mol")

    def spec_conversion(self, other):
        """Return the factor to convert this Units to other with force='spec'."""
        return self.conversion(other, force="spec")


# The conventions for converting between units of different dimensions: each
# treats some physical constants as 1, or a pseudo-dimension as dimensionless.
convention_constants = {
    "spec": ("h", "c"),
    "kBT": ("kB",),
    "mol": ("NA",),
    "mass fraction": ("mass fraction",),
    "volume fraction": ("volume fraction",),
    "amount fraction": ("amount fraction",),
}

# The values (in SI units) and dimensions of these constants, which are
# independent: each has a power of a dimension that the others lack, or of a
# combination of them. A pseudo-dimension is treated as dimensionless by
# treating a constant of value 1 with that dimension as 1.
physical_constants = {
    "h": (h, d_energy * d_time),
    "c": (c, d_length / d_time),
    "kB": (kB, d_energy / d_temperature),
    "NA": (NA, Dimensions(Q=-1)),
    "mass fraction": (1, d_mass_fraction),
    "volume fraction": (1, d_volume_fraction),
    "amount fraction": (1, d_amount_fraction),
}


def powers_of(dims, bases):
    """
    Return the powers p of the independent Dimensions in bases such that
    dims is the product of bases[i]**p[i], or None if there are none. The
    equations, one for each base dimension and pseudo-dimension, are solved
    exactly by Gaussian elimination; the powers are ints where they are
    whole numbers.

    """

    n = len(bases)
    pseudo_names = sorted({name for d in (dims, *bases) for name, _ in d.pseudo})

    def components(d):
        pseudo = dict(d.pseudo)
        return [Fraction(x) for x in d.dims] + [
            Fraction(pseudo.get(name, 0)) for name in pseudo_names
        ]

    # one row of the augmented matrix for each dimension
    columns = [components(base) for base in bases] + [components(dims)]
    rows = [list(row) for row in zip(*columns)]
    for col in range(n):
        pivot = next(r for r in range(col, len(rows)) if rows[r][col] != 0)
        rows[col], rows[pivot] = rows[pivot], rows[col]
        rows[col] = [x / rows[col][col] for x in rows[col]]
        for r in range(len(rows)):
            if r != col and rows[r][col] != 0:
                rows[r] = [x - rows[r][col] * y for x, y in zip(rows[r], rows[col])]
    if any(row[n] != 0 for row in rows[n:]):
        # dims is not a product of powers of the bases
        return None
    return [int(p) if p.denominator == 1 else p for p in (row[n] for row in rows[:n])]


def convert(from_units, to_units, force=None):
    """
    Return the factor required to convert from_units to to_units (Units
    objects or strings), with the conventions in force, as for
    Units.conversion().

    """

    return Units(from_units).conversion(to_units, force=force)
