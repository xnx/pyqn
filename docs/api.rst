*************
API reference
*************

These names can be imported from ``pyqn``. The built-in domains are in
``pyqn.domains``, and listed in the `user guide <guide.html#built-in-domains>`_.

.. currentmodule:: pyqn

Units
=====

.. autoclass:: Units
   :members: parse, conversion, to_si, get_dims, has_units, html, latex

.. autofunction:: convert

Quantities
==========

.. autoclass:: Quantity
   :members: parse, convert_units_to, as_str, value_as_str

Dimensions
==========

.. autoclass:: Dimensions

Domains
=======

.. autoclass:: Domain
   :members: lookup, format

Exceptions
==========

.. autoexception:: UnitsError

.. autoexception:: UnitsConversionError

.. autoexception:: QuantityError
