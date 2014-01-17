# -*- coding: utf-8 -*-
# units.py
# A class representing the units of a physical quantity.
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

import copy
from dimensions import Dimensions
from dimensions import d_dimensionless, d_length, d_energy, d_time
from atom_unit import AtomUnit, UnitsError, feq

class Units(object):
    """
    A class to represent the units of a physical quantity.

    """

    def __init__(self, units):
        """
        Initialize this Units from a list of AtomUnit objects or by parsing
        a string representing the units.

        """

        if type(units) == Units:
            self.__init__(units.atom_units)
        elif type(units) == str:
            self.__init__(self.parse(units).atom_units)
        elif type(units) == list:
            self.atom_units = copy.deepcopy(units)
        else:
            raise TypeError('Attempt to initialize Units object with'
                            ' argument units of type %s' % type(units))
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
    def parse(self, s_compoundunit):
        """
        Parse the string s_compoundunit and return the corresponding
        Units object.

        """

        div_fields = s_compoundunit.split('/')
        ndiv_fields = len(div_fields)
        compound_unit = Units.parse_mult_units(div_fields[0])
        for div_field in div_fields[1:]:
            compound_unit = compound_unit / Units.parse(div_field)
        return compound_unit

    @classmethod
    def parse_mult_units(self, munit):
        """
        Parse a string of units multiplied together (indicated by '.'),
        returning the corresponding Units object.

        """
        atom_units = []
        for s_unit in munit.split('.'):
            atom_unit = AtomUnit.parse(s_unit)
            if atom_unit.base_unit.stem != '1':
                # the unity 'unit' is not really a unit
                atom_units.append(atom_unit)
        return Units(atom_units)

    def _find_atom(self, atom_unit):
        """
        Return the index of atom_unit in the atom_units list, if it exists,
        ignoring any exponent. Otherwise return None.

        """

        for i,my_atom_unit in enumerate(self.atom_units):
            if my_atom_unit.prefix_base_eq(atom_unit):
                return i
        return None

    def __mul__(self, other):
        """ Return the product of this Units object with another. """
        product = Units(self.atom_units)
        for other_atom_unit in other.atom_units:
            i = product._find_atom(other_atom_unit)
            if i is None:
                # new prefix and BaseUnit; append to the resulting product
                product.atom_units.append(other_atom_unit)
            else:
                # this prefix and BaseUnit is already in the AtomUnits
                # of the first operand: update its exponent in the product
                product.atom_units[i].exponent += other_atom_unit.exponent
                product.atom_units[i].dims *= other_atom_unit.dims
                if product.atom_units[i].exponent == 0:
                    # this AtomUnit has cancelled:
                    del product.atom_units[i]
        return product
    def __rmul__(self, other):
        if type(other) == str:
            other = Units(other)
        elif other == 1:
            other = Units('1')
        return self.__mul__(other)

    def __div__(self, other):
        """ Return the ratio of this Units divided by another. """
        if type(other) == str:
            other = Units(other)
        elif other == 1:
            other = Units('1')
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
                ratio.atom_units[i].dims /= other_atom_unit.dims
                if ratio.atom_units[i].exponent == 0:
                    # this AtomUnit has cancelled:
                    del ratio.atom_units[i]
        return ratio
    def __rdiv__(self, other):
        if type(other) == str:
            other = Units(other)
        elif other == 1:
            other = Units('1')
        return other.__div__(self)
 
    def __str__(self):
        """ String representation of this Units. """
        return u'.'.join([unicode(atom_unit) for atom_unit in self.atom_units])

    def __eq__(self, other):
        """ Test for equality with another Units object. """
        if other is None:
            return False
        elif other == 1:
            return self.get_dims() == d_dimensionless
        if self.get_dims() != other.get_dims():
        # obviously the units aren't the same if they have different dimensions
            return False
        # if two Units objects have the same dimensions, they are equal if
        # their conversion factors to SI units are the same: 
        if feq(self.to_si(), other.to_si()):
            return True
        return False

    def __ne__(self, other):
        return not self == other

    def to_si(self):
        """ Return the factor needed to convert this Units to SI. """
        fac = 1.
        for atom_unit in self.atom_units:
            fac *= atom_unit.si_fac
        return fac

    def conversion(self, other, force=None):
        """
        Return the factor required to convert this Units to
        another. Their dimensions have to match, unless force is set
        to allow special cases:
        force='spec':   spectroscopic units: conversions between [L-1],
        [M,L2,T-2], [T-1] are allowed (i.e. cm-1, s-1, and J can be
        interconverted through factors of h, hc).
 
        """

        if type(other) == str:
            other = Units(other)

        if self.get_dims() != other.get_dims():
            if force == 'spec':
                return self.spec_conversion(other)
            raise UnitsError('Failure in units conversion: units %s[%s] and'
                             ' %s[%s] have different dimensions'
                           % (self, self.get_dims(), other, other.get_dims()))
        return self.to_si() / other.to_si()

    def spec_conversion(self, other):
        h = 6.62606896e-34
        c = 299792458.
        d_wavenumber = d_length**-1
        d_frequency = d_time**-1
        from_dims = self.get_dims()
        to_dims = other.get_dims()
        fac = self.to_si()
        if from_dims == d_wavenumber:
            fac *= h*c
        elif from_dims == d_frequency:
            fac *= h
        elif from_dims != d_energy:
            raise UnitsError('Failure in conversion of spectroscopic units:'
                ' I only recognise from-units of wavenumber, energy and'
                ' frequency but got %s' % self.units)
        if to_dims == d_wavenumber:
            fac /= h*c
        elif to_dims == d_frequency:
            fac /= h
        elif to_dims != d_energy:
            raise UnitsError('Failure in conversion of spectroscopic units:'
                ' I only recognise to-units of wavenumber, energy and'
                ' frequency but got %s' % other.units)
        return fac / other.to_si()

