****
PyQn
****

PyQn is a lightweight Python package for parsing, validating, manipulating
and converting physical units and quantities. Units are plain strings in a
compact syntax, such as ``kJ.mol-1``, ``cm-1`` or ``Pa-1/2``; PyQn works out
their dimensions, so conversions and arithmetic are checked, and
*conventions* allow the conversions that are usual practice between units of
different dimensions:

>>> from pyqn import Units
>>> Units("kJ.mol-1").conversion("cm-1", force=("mol", "spec"))
83.59347229111778

*Domains* restrict the units to those of a field, so that unit strings in
data can be checked:

>>> from pyqn import domains
>>> Units("mD", domain=domains.geology)
mdarcy

PyQn is pure Python, with no dependencies, and needs Python 3.10 or later::

    pip install pyqn

.. toctree::
   :maxdepth: 2
   :caption: Contents

   guide

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/spectroscopy
   examples/atmospheric-chemistry
   examples/physical-chemistry
   examples/validating-data

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

The source code is on `GitHub <https://github.com/xnx/pyqn>`_, the package
on `PyPI <https://pypi.org/project/pyqn/>`_, and the changes in each version
are in the `changelog <https://github.com/xnx/pyqn/blob/main/CHANGELOG.md>`_.
