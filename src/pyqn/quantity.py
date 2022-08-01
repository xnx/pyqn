# quantity.py
# A class representing physical quantity, with name, units and uncertainty.
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

import re
import math

# import numpy as np
from .symbol import Symbol
from .units import Units, UnitsError


class QuantityError(Exception):
    """
    An Exception class for errors that might occur whilst manipulating
    Quantity objects.

    """

    def __init__(self, error_str):
        self.error_str = error_str

    def __str__(self):
        return self.error_str


class Quantity(Symbol):
    """
    A Python class representing a physical quantity. This class extends the
    Symbol class, so provides a means of naming the quantity using plain text
    (a utf8-encoded string), LaTeX and HTML, and a definition string.
    A Quantity also has a value, units, and an absolute uncertainty, expressed
    as a standard deviation (sd).

    """

    def __init__(
        self,
        name=None,
        latex=None,
        html=None,
        value=None,
        units=None,
        sd=None,
        definition=None,
    ):
        """
        Initialize the Quantity object: set up its name as for the base,
        Symbol class, and set the quantity's value, sd and units (which
        can be None, a string representing the units, or a Units object and
        will be converted to a Units object in any case).

        """

        Symbol.__init__(self, name, latex, html, definition)
        self.value = value
        self.sd = sd
        if units is None:
            self.units = Units([])
        else:
            self.units = Units(units)

    def __str__(self):
        """A simple string representation of the Quantity."""

        if self.name:
            return "%s = %s %s" % (self.name, self.value, self.units)
        else:
            return "%s %s" % (self.value, self.units)

    __repr__ = __str__

    def value_as_str(self, nsd_digits=2, small=1.0e-3, large=1.0e5):
        """
        Return a string representation of the parameter and its
        standard deviation in the conventional format used in
        the spectroscopic literature:
        Examples:
        0.04857623(71) for 4.857623e-2 +/- 7.1e-7
        -0.412(40) for -0.412 +/- 0.04
        1.7324(38)e7 for 17324000 +/- 380

        Notes:
        This routine has not been rigorously tested, because I wrote
        it at 1am. Therefore, it would be a good idea to output the
        raw parameter values and standard deviations somewhere as
        well as using the output from this routine...

        Arguments:
        nsd_digits: the number of digits in the standard deviation
            (the default is 2)
        small: parameter values smaller than small are output in
            scientific notation (<mantissa>e<power>)
        large: parameter values larger than large are output in
            scientific notation (<mantissa>e-<power>)

        """
        N, sd = self.value, self.sd

        if not sd:
            return str(N)

        absN = abs(N)
        if (absN < small or absN > large) and absN != 0.0:
            # scientific notation
            power = int(math.floor(math.log(absN, 10)))
            N *= pow(10, -power)
            sd *= pow(10, -power)
            dp = int(-math.log(sd, 10)) + nsd_digits
            if dp < 0:
                dp = 0
            sd_digits = int(round(sd * pow(10, dp)))
            fmt = "%." + str(dp) + "f(" + str(sd_digits) + ")e%d"
            return fmt % (N, power)
        else:
            # regular floating point notation
            dp = int(-math.log(sd, 10)) + nsd_digits
            if dp < 0:
                dp = 0
            sd_digits = int(round(sd * pow(10, dp)))
            fmt = "%." + str(dp) + "f(" + str(sd_digits) + ")"

        return fmt % N

    def as_str(self, b_name=True, b_sd=True, b_units=True):
        """
        Output the Quantity object as a string, optionally with name,
        standard deviation and units.

        """

        s = []
        if b_name and self.name:
            s.append("%s = " % self.name)
        if b_sd:
            s.append(self.value_as_str(b_sd))
        else:
            s.append(str(self.value))
        if b_units and self.units.has_units():
            s.append(" %s" % str(self.units))
        return "".join(s)

    def convert_units_to(self, new_units, force=None):
        """
        Convert the Quantity from one set of units to an equivalent set,
        complaining loudly if the two sets of units do not have the same
        dimensions.

        """

        if self.value is None:
            return
        to_units = Units(new_units)
        fac = self.units.conversion(to_units, force)

        #        self.value *= fac
        #        if self.sd is not None:
        #            self.sd *= fac
        #        self.units = to_units
        if self.sd is not None:
            return Quantity(value=self.value * fac, units=new_units, sd=self.sd * fac)
        else:
            return Quantity(value=self.value * fac, units=new_units)

    #    def draw_from_dist(self, shape=None):
    #        """
    #        Return a value or number array of values drawn from the normal
    #        distribution described by this Quantity's mean and standard
    #        deviation. shape is the shape of the NumPy array to return, or
    #        None (the default) to return a single scalar value from the
    #        distribution.
    #
    #        """
    #
    #        if self.sd is None:
    #            raise ValueError('Quantity instance {} has no defined standard'
    #                             ' deviation.'.format(self.name))
    #
    #        return np.random.normal(loc=self.value, scale=self.sd, size=shape)

    def __add__(self, other):
        """
        Add two Quantity objects; they must have the same units. Errors
        are propagated, but assumed to be uncorrelated.

        """

        if type(other) != Quantity:
            raise TypeError
        if self.value is None or other.value is None:
            raise ValueError
        if self.units != other.units:
            raise UnitsError(
                "Can't add two quantities with different"
                " units: %s and %s" % (self.units, other.units)
            )
        if self.sd is None or other.sd is None:
            sd = None
        else:
            sd = math.hypot(self.sd, other.sd)
        return Quantity(value=self.value + other.value, units=self.units, sd=sd)

    def __sub__(self, other):
        """
        Subtract two Quantity objects; they must have the same units. Errors
        are propagated, but assumed to be uncorrelated.

        """

        if type(other) != Quantity:
            raise TypeError
        if self.value is None or other.value is None:
            raise ValueError
        if self.units != other.units:
            raise UnitsError(
                "Can't subtract two quantities with different"
                " units: %s and %s" % (self.units, other.units)
            )
        if self.sd is None or other.sd is None:
            sd = None
        else:
            sd = math.hypot(self.sd, other.sd)
        return Quantity(value=self.value - other.value, units=self.units, sd=sd)

    def __mul__(self, other):
        """
        Multiply this quantity by a number or another quantity. Errors are
        propagated, but assumed to be uncorrelated.

        """

        if self.value is None:
            raise ValueError
        if type(other) in (int, float):
            if self.sd is None:
                sd = None
            else:
                sd = abs(other) * self.sd
            return Quantity(value=self.value * other, units=self.unit, sd=sd)
        else:
            if type(other) != Quantity:
                raise TypeError
            if other.value is None:
                raise ValueError
            value = self.value * other.value
            if not self.sd or not other.sd:
                sd = None
            else:
                sd = value * math.hypot(self.sd / self.value, other.sd / other.value)
            units = self.units * other.units
            return Quantity(value=value, units=units, sd=sd)

    def __truediv__(self, other):
        """
        Divide this quantity by a number or another quantity. Errors are
        propagated, but assumed to be uncorrelated.

        """

        if self.value is None:
            raise ValueError
        if type(other) in (int, float):
            if self.sd is None:
                sd = None
            else:
                sd = abs(other) / self.sd
            return Quantity(value=self.value / other, units=self.units, sd=sd)
        else:
            if type(other) != Quantity:
                raise TypeError
            if other.value is None:
                raise ValueError
            value = self.value / other.value
            if not self.sd or not other.sd:
                sd = None
            else:
                sd = value * math.hypot(self.sd / self.value, other.sd / other.value)
            units = self.units / other.units
            return Quantity(value=value, units=units, sd=sd)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __pow__(self, power):
        return Quantity(value=self.value**power, units=self.units**power)

    @classmethod
    def parse(self, s_quantity, name=None, units=None, sd=None, definition=None):
        s_quantity = s_quantity.strip()
        if "=" in s_quantity:
            fields = s_quantity.split("=")
            s_name = fields[0].strip()
            if s_name:
                name = s_name
            s_quantity = fields[1].strip()
        fields = s_quantity.split(" ")
        s_valsd = fields[0]
        if len(fields) == 2:
            units = Units(fields[1])
        # force lower case, and replace Fortran-style 'D'/'d' exponents
        s_valsd = s_valsd.lower()
        s_valsd = s_valsd.replace("d", "e")
        if "e" in s_valsd:
            s_mantsd, s_exp = s_valsd.split("e")
            exp = int(s_exp)
        else:
            s_mantsd = s_valsd
            exp = 0
        patt = r"([+-]?\d*\.?\d*)\(?(\d+)?\)?"
        m = re.match(patt, s_mantsd)
        if not m:
            raise QuantityError(
                "Failed to parse string into quantity:\n" "%s" % s_mantsd
            )
        s_mantissa, s_sd = m.groups()
        mantissa = float(s_mantissa)
        value = mantissa * 10**exp
        sd = None
        if s_sd:
            if "." in s_mantissa:
                ndp = len(s_mantissa) - s_mantissa.index(".") - 1
            else:
                ndp = 0
            sd = float(s_sd) * 10 ** (exp - ndp)
        return Quantity(name=name, value=value, units=units, sd=sd)
