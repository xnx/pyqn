******************************
Spectroscopy: HITRAN line data
******************************

The HITRAN database gives the parameters of spectral lines in units such as
cm\ :sup:`-1` for line positions and energies, and cm\ :sup:`-1`/(molecule
cm\ :sup:`-2`) for line intensities. The spectroscopy domain has these units,
and the conventions convert between them and the units of other fields.

>>> from pyqn import Quantity, Units, domains
>>> spec = domains.spectroscopy


Line positions
==============

A line position in wavenumbers converts to a frequency with ``force="spec"``,
which treats the speed of light (and Planck's constant) as 1:

>>> nu = Quantity.parse("nu = 2349.143 cm-1", domain=spec)
>>> nu.convert_units_to("THz", force="spec")
70.4255354163494 THz

A wavelength is inversely proportional to the wavenumber, rather than a
multiple of it, so pyqn does not convert wavenumbers to wavelengths.


Lower-state energies as temperatures
====================================

An energy in cm\ :sup:`-1` converts to a temperature with the conventions
``spec`` and ``kBT`` together. The factor for 1 cm\ :sup:`-1` is the second
radiation constant, c\ :sub:`2` = hc/k\ :sub:`B`, in cm K:

>>> Units("cm-1").conversion("K", force=("spec", "kBT"))
1.4387768775039338
>>> E = Quantity.parse("E = 1175.7603 cm-1", domain=spec)
>>> E.convert_units_to("K", force=("spec", "kBT"))
1691.6567331270883 K


Line intensities
================

pyqn reads HITRAN's units for a line intensity as written, because each
``/`` divides by everything up to the next one; they simplify to cm
molec\ :sup:`-1`. A molecule, ``molec``, is a unit of amount (exactly
1/N\ :sub:`A` mol), so an intensity converts to the molar units of band
intensities directly:

>>> S = Quantity.parse("S = 3.521(35)e-18 cm-1/molec.cm-2", domain=spec)
>>> S.units
cm.molec-1
>>> S.convert_units_to("km.mol-1").as_str()
'21.20(21) km.mol-1'


Line widths
===========

HITRAN's air-broadened half-widths are in cm\ :sup:`-1` atm\ :sup:`-1`. They
convert to MHz Torr\ :sup:`-1` with ``force="spec"``, because the
conventions apply to units of any dimensions:

>>> gamma = Quantity.parse("gamma = 0.0700(7) cm-1.atm-1", domain=spec)
>>> gamma.convert_units_to("MHz.Torr-1", force="spec").as_str()
'2.761(28) MHz.Torr-1'

The half-width at a particular pressure is a product of quantities. The
pressure is given a standard deviation of 0, so that it counts as exact;
without one, the uncertainty of the product would be unknown:

>>> hwhm = gamma * Quantity(value=0.5, units="atm", sd=0)
>>> hwhm.as_str()
'0.03500(35) cm-1'
>>> hwhm.convert_units_to("MHz", force="spec").as_str()
'1049(10) MHz'
