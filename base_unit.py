# -*- coding: utf-8 -*-

# Christian Hill
# v0.2 21/11/2012
# v0.1 28/11/2011
#
# The BaseUnit class, representing a "base" unit to a physical quantity.

from dimensions import *

class BaseUnit(object):
    """
    A BaseUnit is a commonly-used single unit without a prefix, for example:
    m (metre, length), erg (energy), bar (pressure). BaseUnit objects have
    a description and know their dimensions in terms of powers of Length,
    Time, Mass, etc. (described by a Dimensions object). 

    """

    def __init__(self, stem, name, unit_type, fac, description, latex,
                 dims=None):
        """ Initialize the BaseUnit object. """
        # BaseUnit stem, e.g. 'm', 'g', 'bar', ...
        self.stem = stem
        # BaseUnit name, e.g. 'metre', 'gram', 'bar', ...
        self.name = name
        # BaseUnit type, e.g. 'length', 'mass', 'pressure', ...
        self.unit_type = unit_type
        # fac is the relationship between this base unit and the SI base
        # unit, e.g. m: 1. (m is SI), g: 1.e-3 (1 g = 1.e-3 kg),
        # bar: 1.e5 (1 bar = 10e5 Pa)
        self.fac = fac
        # any further description of the BaseUnit
        self.description = description
        # LaTex representation of the BaseUnit
        self.latex = latex
        # BaseUnit dimensions, as a Dimension object
        self.dims = dims

    def __eq__(self, other):
        if self.stem == other.stem:
            return True

    def __str__(self):
        """ String representation of the BaseUnit is just its 'stem'. """
        return self.stem

base_units = [

BaseUnit('1', 'unity', 'unity', 1., '', '1', d_dimensionless),

BaseUnit('m', 'metre', 'length', 1., '', 'm', d_length),
BaseUnit('s', 'second', 'time', 1., '', 's', d_time),
BaseUnit('g', 'gram', 'mass', 1.e-3, '', 'g', d_mass),
BaseUnit('K', 'kelvin', 'temperature', 1., '', 'K', Dimensions(Theta=1)),
BaseUnit('mol', 'mole', 'amount', 1., '', 'mol', Dimensions(Q=1)),
BaseUnit('N', 'newton', 'force', 1., '', 'N', d_force),
BaseUnit('J', 'joule', 'energy', 1., '', 'J', d_energy),
BaseUnit('W', 'watt', 'power', 1., '', 'W', d_energy / d_time),
BaseUnit('Pa', 'pascal', 'pressure', 1., '', 'Pa', d_pressure),
BaseUnit('C', 'coulomb', 'charge', 1., '', 'C', d_charge),
BaseUnit('A', 'amp', 'current', 1., '', 'A', Dimensions(C=1)),
BaseUnit('V', 'volt', 'voltage', 1., '', 'V', d_voltage),
BaseUnit('T', 'tesla', 'magnetic field strength', 1., '', 'T',
         d_magfield_strength),
BaseUnit('Hz', 'hertz', 'cyclic frequency', 1., '', 'Hz', d_time**-1),

# Dimensionless units
BaseUnit('deg', 'degree', 'angle', 0.017453292519943295, '', 'deg',
         d_dimensionless),
BaseUnit('rad', 'radian', 'angle', 1., '', 'rad', d_dimensionless),
BaseUnit('arcmin', 'arcminute', 'angle', 2.908882086657216e-4, '', 'arcmin',
         d_dimensionless),
# NB we can't allow as for arcseconds because of ambiguity with attoseconds
BaseUnit('asec', 'arcsecond', 'angle', 4.84813681109536e-6, '', 'asec',
         d_dimensionless),
BaseUnit('sr', 'steradian', 'solid angle', 1., '', 'sr', d_dimensionless),

# Non-SI pressure units
BaseUnit('bar', 'bar', 'pressure', 1.e5, '', 'bar', d_pressure),
BaseUnit('atm', 'atmosphere', 'pressure', 1.01325e5, '', 'atm', d_pressure),
BaseUnit('Torr', 'torr', 'pressure', 133.322368, '', 'Torr', d_pressure),
# (but see e.g. Wikipedia for the precise relationship between Torr and mmHg
BaseUnit('mmHg', 'millimetres of mercury', 'pressure', 133.322368, '', 'mmHg',
         d_pressure),

# Non-SI force units
BaseUnit('dyn', 'dyne', 'force', 1.e-5, '', 'dyn', d_force),
# Energy units
BaseUnit('erg', 'erg', 'energy', 1.e-7, '', 'erg', d_energy),
BaseUnit('eV', 'electron volt', 'energy',  1.602176487e-19, '', 'eV',
         d_energy),
BaseUnit('E_h', 'hartree', 'energy', 4.35974394e-18, '', 'E_h', d_energy),
BaseUnit('cal', 'thermodynamic calorie', 'energy', 4.184, '', 'cal', d_energy),
BaseUnit('Ry', 'rydberg', 'energy', 13.60569253 * 1.602176487e-19, '', 'Ry',
         d_energy),

# Non-SI mass units
BaseUnit('u', 'atomic mass unit', 'mass', 1.660538921e-27, '', 'u', d_mass),
BaseUnit('amu', 'atomic mass unit', 'mass', 1.660538921e-27, '', 'amu', d_mass),
BaseUnit('Da', 'dalton', 'mass', 1.660538921e-27, '', 'Da', d_mass),
BaseUnit('m_e', 'electron mass', 'mass', 9.10938291e-31, '', 'm_e', d_mass),

# Non-SI length units
BaseUnit(u'Å', 'angstrom', 'length', 1.e-10, '', '\AA', d_length),
BaseUnit('a0', 'bohr', 'length', 5.2917721092e-11, '', 'a_0', d_length),

# Non-SI area units
BaseUnit('b', 'barn', 'area', 1.e-28, '', 'b', d_area),

# Non-SI volume units
BaseUnit('l', 'litre', 'volume', 1.e-3, '', 'l', d_volume),

# Non-SI time units
BaseUnit('min', 'minute', 'time', 60., '', 'min', d_time),
BaseUnit('hr', 'hour', 'time', 3600., '', 'hr', d_time),
BaseUnit('h', 'h', 'time', 3600., '', 'h', d_time),
BaseUnit('d', 'day', 'time', 86400., '', 'd', d_time),

# Other cgs units
BaseUnit('k', 'kayser', 'wavenumber', 100., '', 'k', d_length**-1),
BaseUnit('D', 'debye', 'electric dipole moment', 1.e-21/299792458., '', 'D',
         d_charge * d_length),

BaseUnit('hbar', 'hbar', 'angular momentum', 1.05457148e-34, '', '\hbar',
         Dimensions(L=2, M=1, T=-1)),
BaseUnit('e', 'electron charge', 'charge', 1.602176565e-19, '', 'e', d_charge),

]

# create a dictionary mapping the BaseUnit stems (as keys) to the BaseUnit
# objects (as values):
base_unit_stems = {}
for base_unit in base_units:
    base_unit_stems[base_unit.stem] = base_unit
