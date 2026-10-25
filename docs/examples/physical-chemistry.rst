**********************************************************
Physical chemistry: energies, entropies and concentrations
**********************************************************

The chemistry domain has the units that chemists use with the SI units, such
as ``kcal``, ``M`` (molar), ``Torr`` and ``Å``. The conventions convert
between molar energies and energies per molecule, wavenumbers and
temperatures.

>>> from pyqn import Quantity, convert, domains
>>> chem = domains.chemistry


Molar energies
==============

``kcal.mol-1`` has the same dimensions as ``kJ.mol-1``, so it needs no
convention; the others need the conventions that treat the Avogadro constant
(``mol``), Planck's constant and the speed of light (``spec``) and the
Boltzmann constant (``kBT``) as 1:

>>> for units, force in (
...     ("kcal.mol-1", None),
...     ("eV", "mol"),
...     ("cm-1", ("mol", "spec")),
...     ("K", ("mol", "kBT")),
... ):
...     print(f"1 kJ/mol = {convert('kJ.mol-1', units, force=force):.6g} {units}")
1 kJ/mol = 0.239006 kcal.mol-1
1 kJ/mol = 0.0103643 eV
1 kJ/mol = 83.5935 cm-1
1 kJ/mol = 120.272 K

With ``mol`` and ``kBT`` together, a temperature converts to a molar energy
through the gas constant, R = N\ :sub:`A` k\ :sub:`B`:

>>> RT = 298.15 * convert("K", "kJ.mol-1", force=("mol", "kBT"))
>>> print(f"RT = {RT:.5f} kJ/mol at 298.15 K")
RT = 2.47896 kJ/mol at 298.15 K

Quantities carry their uncertainties through conversions:

>>> dH = Quantity.parse("dH = -285.83(4) kJ.mol-1", domain=chem)
>>> dH.convert_units_to("kcal.mol-1").as_str()
'-68.3150(96) kcal.mol-1'
>>> dH.convert_units_to("eV", force="mol").as_str()
'-2.96242(41) eV'


Entropies in units of R
=======================

The same conventions turn a molar entropy or heat capacity into a number of
gas constants:

>>> S_H2 = Quantity.parse("S = 130.68(1) J.K-1.mol-1", domain=chem)
>>> S_H2.convert_units_to("1", force=("mol", "kBT")).as_str()
'15.7172(12)'


Concentrations
==============

The molar, ``M``, is a mole per litre, and a molecule is a unit of amount, so
concentrations and numbers of molecules convert directly:

>>> convert("mM", "mol.m-3")
1.0
>>> print(f"1 μM = {convert('μM', 'molec.μL-1'):.4g} molecules per μL")
1 μM = 6.022e+11 molecules per μL
>>> c = Quantity(value=2.5, units="mM", domain=chem)
>>> V = Quantity(value=10, units="mL", domain=chem)
>>> (c * V).convert_units_to("μmol")
25.0 μmol
