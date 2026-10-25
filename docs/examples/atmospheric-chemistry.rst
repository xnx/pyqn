***********************************************************
Atmospheric chemistry: mixing ratios and number densities
***********************************************************

The meteorology domain has the units of pressure that meteorologists use,
such as ``hPa`` and ``mbar``, volume mixing ratios (``ppmv``, ``ppbv``,
``pptv``) and the molecule.

>>> from pyqn import Quantity, Units, domains
>>> met = domains.meteorology
>>> Units("hPa", domain=met) == Units("mbar", domain=met)
True


Mixing ratios
=============

A volume mixing ratio is a volume fraction, and its units have the
pseudo-dimension *volume fraction*, so it converts to other volume
fractions:

>>> x_co2 = Quantity.parse("x = 420.0 ppmv", domain=met)
>>> x = x_co2.convert_units_to("ppbv")
>>> print(f"{x.value:.6g} {x.units}")
420000 ppbv

For an ideal gas, a volume fraction is the same as an amount fraction, in
mol mol\ :sup:`-1`. pyqn only treats the volume fraction as dimensionless if
you say so:

>>> x_co2.convert_units_to("μmol/mol", force="volume fraction")
420.0 μmol.mol-1


Number densities
================

The number density of a gas is n = p/(k\ :sub:`B`\ T), which pyqn can
calculate with quantities. The result, in hPa J\ :sup:`-1`, is a number of
molecules per volume, and ``force="mol"`` counts it in molecules:

>>> p = Quantity(value=1013.25, units="hPa")
>>> T = Quantity(value=288.15, units="K")
>>> kB = Quantity(value=1.380649e-23, units="J.K-1")
>>> n_air = (p / (kB * T)).convert_units_to("molec.cm-3", force="mol")
>>> print(f"{n_air.value:.4e} {n_air.units}")
2.5469e+19 molec.cm-3

The number density of a trace gas is its mixing ratio times that of the air:

>>> n_co2 = (x_co2 * n_air).convert_units_to("molec.cm-3", force="volume fraction")
>>> print(f"{n_co2.value:.4e} {n_co2.units}")
1.0697e+16 molec.cm-3


Mass mixing ratios
==================

A volume fraction and a mass fraction cannot be converted into each other
without the molar masses, so pyqn refuses:

>>> x_co2.convert_units_to("ppmw")
Traceback (most recent call last):
    ...
UnitsError: Failure in units conversion: units ppmv ([volume fraction]) and ppmw ([mass fraction]) have different dimensions

The mass mixing ratio of CO\ :sub:`2` in air is w = x M\ :sub:`CO2` /
M\ :sub:`air`. Having done that physics, say so with both conventions:

>>> M_co2 = Quantity(value=44.009, units="g.mol-1")
>>> M_air = Quantity(value=28.964, units="g.mol-1")
>>> w_co2 = (x_co2 * M_co2 / M_air).convert_units_to(
...     "ppmw", force=("volume fraction", "mass fraction")
... )
>>> print(f"{w_co2.value:.1f} {w_co2.units}")
638.2 ppmw

The meteorology domain has no mass fractions, so it rejects them:

>>> Units("ppmw", domain=met)
Traceback (most recent call last):
    ...
UnitsError: Unrecognised unit in the meteorology domain: ppmw
