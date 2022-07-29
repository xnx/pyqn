********************
Introduction to PyQn
********************



PyQn is a Python package for parsing, validating, manipulating and
transforming physical quantities and their units.

Units are specified as strings using a simple and flexible syntax,
and may be compared, output in different formats and manipulated using a
variety of predefined Python methods.



Installation:
=============

The PyQn package can be installed either from PyPI_ using pip

.. code-block:: bash

    python3 -m pip install pyqn

or from the source by running (one of the two) from the project source directory.

.. code-block:: bash

    # either
    python setup.py install

    # or
    python3 -m pip install .



Examples:
=========

Units
-----
The units of physical quantities are represented by the ``Units`` class. A
``Units`` object is instantiated from a valid units string and supports ions,
isotopologues, as well as a few special species. This object contains
attributes including the dimensions, HTML and LaTeX representations, and
methods for conversion to different compatible units.

.. code-block:: pycon

    >>> from pyqn.units import Units
    >>> u1 = Units('km')

    >>> u2 = Units('hr')

    >>> u3 = u1/u2

    >>> print(u3)
    km.hr-1

    >>> u4 = Units('m/s')

    >>> u3.conversion(u4)        # OK: can convert from km/hr to m/s
    Out[7]: 0.2777777777777778

    >>> u3.conversion(u2)        #  Oops: can't convert from km/hr to m!
    ...
    UnitsError: Failure in units conversion: units km.hr-1[L.T-1] and hr[T] have
    different dimensions
