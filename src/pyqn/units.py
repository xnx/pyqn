# units.py
# A class representing the units of a physical quantity.
#
# Copyright (C) 2012-2016 Christian Hill
# SPDX-License-Identifier: GPL-3.0-or-later

import copy
import math
import re
import unicodedata

from .atom_unit import AtomUnit, UnitsError
from .constants import NA, c, h, kB
from .dimensions import Dimensions, d_energy, d_length, d_temperature, d_time


class UnitsConversionError(UnitsError):
    pass


class Units:
    """
    A class to represent the units of a physical quantity.

    """

    def __init__(self, units):
        """
        Initialize this Units from a list of AtomUnit objects or by parsing
        a string representing the units.

        """

        self.undef = False
        self.dims = None

        if type(units) is Units:
            if units.undef:
                self.undef = True
                return
            self.__init__(units.atom_units)
        elif type(units) is str:
            if units == "undef":
                self.undef = True
                return
            self.__init__(self.parse(units).atom_units)
        elif type(units) is list:
            self.atom_units = copy.deepcopy(units)
        else:
            raise TypeError(
                "Attempt to initialize Units object with"
                f" argument units of type {type(units)}"
            )
        # also get the dimensions of the units
        self.dims = self.get_dims()

    def has_units(self):
        if self.atom_units:
            return True
        return False

    def get_dims(self):
        """
        Return a Dimensions object representing the dimensions of this
        Units.

        """

        dims = Dimensions()
        for atom_unit in self.atom_units:
            dims *= atom_unit.dims
        return dims

    @classmethod
    def parse(self, s_compoundunit, no_divided_units=False):
        """
        Parse the string s_compoundunit and return the corresponding
        Units object.

        """

        # Read look-alike characters such as the micro sign (U+00B5) and the
        # ohm sign (U+2126) as the characters in the unit table (U+03BC, U+03A9).
        s_compoundunit = unicodedata.normalize("NFKC", s_compoundunit)
        if re.search(r"\s", s_compoundunit):
            raise UnitsError(f"Units cannot contain spaces: {s_compoundunit!r}")

        if no_divided_units:
            # s_compoundunit does not consist of any units separated by '/'
            # so parse immediately as a sequence of multiplied units.
            return Units.parse_mult_units(s_compoundunit)

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
        compound_unit = Units.parse_mult_units(div_fields[0].replace(":", "/"))
        for div_field in div_fields[1:]:
            div_field = div_field.replace(":", "/")
            compound_unit = compound_unit / Units.parse(
                div_field, no_divided_units=True
            )
        return compound_unit

    @classmethod
    def parse_mult_units(self, munit):
        """
        Parse a string of units multiplied together (indicated by '.'),
        returning the corresponding Units object.

        """
        atom_units = []
        for s_unit in munit.split("."):
            atom_unit = AtomUnit.parse(s_unit)
            if atom_unit.base_unit.stem != "1":
                # the unity 'unit' is not really a unit
                atom_units.append(atom_unit)
        return Units(atom_units)

    def _find_atom(self, atom_unit):
        """
        Return the index of atom_unit in the atom_units list, if it exists,
        ignoring any exponent. Otherwise return None.

        """

        for i, my_atom_unit in enumerate(self.atom_units):
            if my_atom_unit.prefix_base_eq(atom_unit):
                return i
        return None

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

        product = Units(self.atom_units)
        for other_atom_unit in other.atom_units:
            i = product._find_atom(other_atom_unit)
            if i is None:
                # new prefix and BaseUnit; append (a copy) to the product
                product.atom_units.append(copy.copy(other_atom_unit))
            else:
                # this prefix and BaseUnit is already in the AtomUnits
                # of the first operand: update its exponent in the product
                product.atom_units[i].exponent += other_atom_unit.exponent
                if product.atom_units[i].exponent == 0:
                    # this AtomUnit has cancelled:
                    del product.atom_units[i]
        product.dims = product.get_dims()
        return product

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

        ratio = Units(self.atom_units)
        for other_atom_unit in other.atom_units:
            i = ratio._find_atom(other_atom_unit)
            if i is None:
                # new prefix and BaseUnit; append to the resulting ratio
                ratio.atom_units.append(other_atom_unit**-1)
            else:
                # this prefix and BaseUnit is already in the AtomUnits
                # of the first operand: update its exponent in the ratio
                ratio.atom_units[i].exponent -= other_atom_unit.exponent
                if ratio.atom_units[i].exponent == 0:
                    # this AtomUnit has cancelled:
                    del ratio.atom_units[i]
        ratio.dims = ratio.get_dims()
        return ratio

    def __rtruediv__(self, other):
        other = self._as_units(other)
        if other is None:
            return NotImplemented
        return other / self

    def __pow__(self, power):
        if power == 0:
            return Units([])
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

    def to_si(self):
        """Return the factor needed to convert this Units to SI."""
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
        Return the factor required to convert this Units to
        another. Their dimensions have to match, unless force is set
        to allow special cases:
        force='spec':   spectroscopic units: conversions between [L-1],
        [M,L2,T-2], [T-1] are allowed (i.e. cm-1, s-1, and J can be
        interconverted through factors of h, hc).

        """

        if isinstance(other, str):
            other = Units(other)

        conversion_method = {
            "spec": self.spec_conversion,
            "mol": self.mol_conversion,
            "kBT": self.kBT_conversion,
        }

        self_dims, other_dims = self.get_dims(), other.get_dims()
        if self_dims != other_dims:
            try:
                return conversion_method[force](other)
            except KeyError:
                raise UnitsError(
                    f"Failure in units conversion: units {self}[{self.get_dims()}] and"
                    f" {other}[{other.get_dims()}] have different dimensions"
                )
        return self.to_si() / other.to_si()

    def kBT_conversion(self, other):
        from_dims = self.get_dims()
        to_dims = other.get_dims()
        fac = self.to_si()

        if from_dims == d_energy and to_dims == d_temperature:
            fac = fac / kB
        elif from_dims == d_temperature and to_dims == d_energy:
            fac = fac * kB
        else:
            raise UnitsError(
                "Failure in conversion of units: was expecting to "
                "convert between energy and temperature"
            )
        return fac / other.to_si()

    def mol_conversion(self, other):
        from_dims, to_dims = self.get_dims(), other.get_dims()
        from_dims_Q, to_dims_Q = from_dims.dims[4], to_dims.dims[4]

        # We can only remove or add the amount dimension in its entirety, and
        # the other dimensions must match.
        if (from_dims_Q and to_dims_Q) or (
            from_dims / Dimensions(Q=from_dims_Q) != to_dims / Dimensions(Q=to_dims_Q)
        ):
            raise UnitsConversionError(
                f"Cannot force molar conversion between {self} and {other}"
            )

        fac = self.to_si()
        if from_dims_Q:
            fac = fac * NA**from_dims_Q
        else:
            fac = fac * NA**-to_dims_Q
        return fac / other.to_si()

    def spec_conversion(self, other):
        d_wavenumber = d_length**-1
        d_frequency = d_time**-1

        from_dims = self.get_dims()
        to_dims = other.get_dims()
        fac = self.to_si()
        if from_dims == d_wavenumber:
            fac *= h * c
        elif from_dims == d_frequency:
            fac *= h
        elif from_dims != d_energy:
            raise UnitsError(
                "Failure in conversion of spectroscopic units:"
                " I only recognise from-units of wavenumber, energy and"
                f" frequency but got {self}"
            )
        if to_dims == d_wavenumber:
            fac /= h * c
        elif to_dims == d_frequency:
            fac /= h
        elif to_dims != d_energy:
            raise UnitsError(
                "Failure in conversion of spectroscopic units:"
                " I only recognise to-units of wavenumber, energy and"
                f" frequency but got {other}"
            )
        return fac / other.to_si()


def convert(from_units, to_units):
    return Units(from_units).conversion(to_units)
