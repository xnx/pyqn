# -*- coding: utf-8 -*-
# atom_unit.py

# Christian Hill, 29/3/13
# Department of Physics and Astronomy, University College London
# christian.hill@ucl.ac.uk

import sys
from pyparsing import Word, Group, Literal, Suppress, ParseException, oneOf,\
                      Optional
from si import si_prefixes
from base_unit import BaseUnit, base_unit_stems
from dimensions import Dimensions

# pyparsing stuff for parsing unit strings:
caps = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
lowers = caps.lower()
letters = u'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_0'
digits = u'123456789'
exponent = Word(digits + '-')
prefix = oneOf(si_prefixes.keys())
ustem = Word(letters + u'Å')
uatom = (Group( u'1' | (Optional(prefix) + ustem)) + Optional(exponent))\
            | (Group( u'1' | ustem) + Optional(exponent))

# floating point equality and its negation, to some suitable tolerance
def feq(f1, f2, tol=1.e-10):
    return abs(f1-f2) <= tol
def fneq(f1, f2, tol=1.e-10):
    return not feq(f1, f2, tol)

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
        """ Initialize the AtomUnit object. """

        self.base_unit = base_unit
        self.exponent = exponent
        self.dims = self.base_unit.dims ** self.exponent

        # get the SI prefix (if present), and its 'factor' (10 raised to its
        # the power it represents
        self.si_fac = 1.
        self.prefix = prefix
        if prefix is not None:
            try:
                self.si_prefix = si_prefixes[prefix]
            except KeyError:
                raise UnitsError('Invalid or unsupported SI prefix: %s'
                                                        % prefix)
            self.si_fac = self.si_prefix.fac
        # now calculate the factor relating this AtomUnit to its
        # corresponding SI unit:
        self.si_fac = (self.si_fac * self.base_unit.fac) ** self.exponent

    @classmethod
    def parse(self, s_unit_atom):
        """
        Parse the string s_unit_atom into an AtomUnit object and return
        it.

        """

        try:
            uatom_data = uatom.parseString(s_unit_atom)
        except ParseException:
            raise
            raise UnitsError("Invalid unit atom syntax: %s" % s_unit_atom)

        # uatom_data comes back as (([prefix], <stem>), [exponent])
        if len(uatom_data[0]) == 1:
            # no prefix, just the stem
            prefix = None
            stem = uatom_data[0][0]
        else:
            prefix = uatom_data[0][0]
            stem = uatom_data[0][1]
            # we need to ensure that units such as 'mmHg' get resolved
            # properly (ie to mmHg, not to milli-mHg):
            if stem not in base_unit_stems:
                prefix = None
                stem = ''.join(uatom_data[0])
        try:
            base_unit = base_unit_stems[stem]
        except:
            raise UnitsError("Unrecognised unit: %s" % unit_atom)

        # if there is an exponent, determine what it is (default is 1)
        exponent = 1
        if len(uatom_data) == 2:
            exponent = int(uatom_data[1])
        return AtomUnit(prefix, base_unit, exponent)

    def __pow__(self, power):
        """ Return the current AtomUnit raised to a specified power. """
        return AtomUnit(self.prefix, self.base_unit, self.exponent * power)

    def __str__(self):
        """ String representation of this AtomUnit. """
        s = ''
        if self.prefix:
            s = self.prefix
        s_exponent = ''
        if self.exponent != 1:
            s_exponent = unicode(self.exponent)
        return ''.join([s, unicode(self.base_unit), s_exponent])

    def prefix_base_eq(self, other):
        """
        Compare two AtomUnits and return true if they have the same BaseUnit
        and the same prefix.

        """

        if self.prefix == other.prefix and self.base_unit == other.base_unit:
            return True
        return False

