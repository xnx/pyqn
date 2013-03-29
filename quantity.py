# -*- coding: utf-8 -*-
# quantity.py

# Christian Hill, 29/3/13
# Department of Physics and Astronomy, University College London
# christian.hill@ucl.ac.uk

import re
import math
from symbol import Symbol
from units import Units, UnitsError

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

    def __init__(self, name=None, latex=None, html=None, value=None,
                 units=None, sd=None, definition=None):
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
        """ A simple string representation of the Quantity. """

        if self.name:
            return '%s = %s %s' % (self.name, self.value, self.units)
        else:
            return '%s %s' % (self.value, self.units)

    def convert_units_to(self, new_units):
        """
        Convert the Quantity from one set of units to an equivalent set,
        complaining loudly if the two sets of units do not have the same
        dimensions.

        """

        if self.value is None:
            return
        to_units = Units(new_units)
        try:
            assert to_units.get_dims() == self.units.get_dims()
        except AssertionError:
            raise UnitsError('Can\'t convert between units with different'
                  ' dimensions')
        fac = self.units.conversion(to_units)
        self.value *= fac
        if self.sd is not None:
            self.sd *= fac
        self.units = to_units
        return

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
            raise UnitsError('Can\'t add two quantities with different'
                             ' units: %s and %s' % (self.units, other.units))
        if self.sd is None or other.sd is None:
            sd = None
        else:
            sd = math.hypot(self.sd, other.sd)
        return Quantity(value=self.value+other.value, units=self.units, sd=sd)

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
            raise UnitsError('Can\'t subtract two quantities with different'
                             ' units: %s and %s' % (self.units, other.units))
        if self.sd is None or other.sd is None:
            sd = None
        else:
            sd = math.hypot(self.sd, other.sd)
        return Quantity(value=self.value-other.value, units=self.units, sd=sd)

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
            return Quantity(value=self.value*other, units=self.unit, sd=sd)
        else:
            if type(other) != Quantity:
                raise TypeError
            if other.value is None:
                raise ValueError
            value = self.value * other.value
            if not self.sd or not other.sd:
                sd = None
            else:
                sd = value * math.hypot(self.sd/self.value,
                                        other.sd/other.value)
            units = self.units * other.units
            return Quantity(value=value, units=units, sd=sd)

    @classmethod
    def parse(self, s_quantity, name=None, units=None, sd=None,
              definition=None):
        s_quantity = s_quantity.strip()
        if '=' in s_quantity:
            fields = s_quantity.split('=')
            s_name = fields[0].strip()
            if s_name:
                name = s_name
            s_quantity = fields[1].strip()
        fields = s_quantity.split(' ')
        s_valsd = fields[0]
        if len(fields) == 2:
            units = Units(fields[1])
        # force lower case, and replace Fortran-style 'D'/'d' exponents
        s_valsd = s_valsd.lower()
        s_valsd = s_valsd.replace('d','e')
        if 'e' in s_valsd:
            s_mantsd, s_exp = s_valsd.split('e')
            exp = int(s_exp)
        else:
            s_mantsd = s_valsd
            exp = 0
        patt = '([+-]?\d*\.?\d*)\(?(\d+)?\)?'
        m = re.match(patt, s_mantsd)
        if not m:
            raise QuantityError('Failed to parse string into quantity:\n'\
                                '%s' % s_mantsd)
        s_mantissa, s_sd = m.groups()
        mantissa = float(s_mantissa)
        value = mantissa * 10**exp
        sd = None
        if s_sd:
            if '.' in s_mantissa:
                ndp = len(s_mantissa) - s_mantissa.index('.') - 1
            else:
                ndp = 0
            sd = float(s_sd) * 10**(exp-ndp)
        return Quantity(name=name, value=value, units=units, sd=sd)
