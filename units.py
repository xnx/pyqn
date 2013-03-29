# -*- coding: utf-8 -*-
# units.py

# Christian Hill, 29/3/13
# Department of Physics and Astronomy, University College London
# christian.hill@ucl.ac.uk

import copy
from dimensions import Dimensions
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
            raise TypeError
        # also get the dimensions of the units
        self.dims = self.get_dims()

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
                if product.atom_units[i].exponent == 0:
                    # this AtomUnit has cancelled:
                    del product.atom_units[i]
        return product

    def __div__(self, other):
        """ Return the ratio of this Units divided by another. """
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
        return ratio
 
    def __str__(self):
        """ String representation of this Units. """
        return u'.'.join([unicode(atom_unit) for atom_unit in self.atom_units])

    def __eq__(self, other):
        """ Test for equality with another Units object. """
        if other is None:
            return False
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

    def conversion(self, other):
        """
        Return the factor required to convert this Units to
        another. Their dimensions have to match, of course.
 
        """

        if self.get_dims() != other.get_dims():
            raise UnitsError('Failure in units conversion: units %s[%s] and'
                             ' %s[%s] have different dimensions'
                           % (self, self.get_dims(), other, other.get_dims()))
        return self.to_si() / other.to_si()
