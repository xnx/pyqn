*************************************
Validating unit strings in data files
*************************************

A unit string that parses isn't necessarily a unit that makes sense in your
field. Parsing a data file's unit strings in a domain rejects those that
don't, and checking their dimensions catches values given in the wrong kind
of units.

>>> import csv
>>> import io
>>> from pyqn import Domain, Units, UnitsError, domains


Checking unit strings
=====================

Here is a small data file, as a string, with some problems in its units:

>>> data = io.StringIO("""quantity,value,units
... wavenumber,2349.1,cm-1
... intensity,3.5e-19,cm-1/molec.cm-2
... width,0.07,cm-1 atm-1
... lower-state energy,1691.66,K
... distance,3,furlong
... pressure,1,mmHg
... """)
>>> rows = list(csv.DictReader(data))
>>> for row in rows:
...     try:
...         units = Units(row["units"], domain=domains.spectroscopy)
...     except UnitsError as error:
...         print(f"{row['quantity']}: {error}")
width: Units cannot contain spaces: 'cm-1 atm-1'
distance: Unrecognised unit in the spectroscopy domain: furlong
pressure: Unrecognised unit in the spectroscopy domain: mmHg

If your data uses units that the built-in domain lacks, such as ``mmHg``,
make a domain of your own from it:

>>> lab = Domain(
...     "lab",
...     units=[*domains.spectroscopy.units, domains.everything.stems["mmHg"]],
...     spellings=dict(domains.spectroscopy.spellings),
... )
>>> Units("mmHg", domain=lab)
mmHg


Checking dimensions
===================

A unit string can be valid but have the wrong dimensions for its quantity.
Comparing the dimensions with those of the expected units catches that:

>>> expected = {"wavenumber": "cm-1", "lower-state energy": "cm-1"}
>>> for row in rows:
...     if row["quantity"] in expected:
...         units = Units(row["units"], domain=lab)
...         if units.dims != Units(expected[row["quantity"]]).dims:
...             print(f"{row['quantity']}: {units} ({units.dims}) is not like"
...                   f" {expected[row['quantity']]}")
lower-state energy: K (Theta) is not like cm-1

Here, the conventions can convert the value, if that is what was meant:

>>> energy = 1691.66 * Units("K").conversion("cm-1", force=("kBT", "spec"))
>>> print(f"{energy:.4f} cm-1")
1175.7626 cm-1


Normalising unit strings
========================

The string that pyqn prints for units is their canonical form, which means
the same in every domain, so it can be stored and read back reliably:

>>> str(Units("cm-1/molec.cm-2", domain=domains.spectroscopy))
'cm.molec-1'
>>> str(Units("J/mol.K"))
'J.mol-1.K-1'


Deprecated prefixes
===================

In the default domain, prefixes that pyqn 2.0 accepted on units that no
longer take them, such as the ``k`` of ``kmin``, give a
``DeprecationWarning``. To treat these as errors while validating data, turn
the warnings into exceptions:

>>> import warnings
>>> with warnings.catch_warnings():
...     warnings.simplefilter("error")
...     Units("kmin")
Traceback (most recent call last):
    ...
DeprecationWarning: kmin: the unit min will not take the prefix k in pyqn 3.0
