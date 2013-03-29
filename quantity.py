# -*- coding: utf-8 -*-
# quantity.py

# Christian Hill, 29/3/13
# Department of Physics and Astronomy, University College London
# christian.hill@ucl.ac.uk

import math
from symbol import Symbol
from units import Units, UnitsError

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

