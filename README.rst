****
PyQn
****

PyQn is a lightweight Python package for parsing, validating, manipulating
and converting physical units and quantities.

Units are written as plain strings in a compact syntax, such as
``kJ.mol-1``, ``cm-1`` or ``Pa-1/2``. PyQn works out their dimensions, so
conversions and arithmetic between them are checked. A few *conventions*
allow conversions between units of different dimensions where that is usual
practice, such as between wavenumbers and energies in spectroscopy.

PyQn is pure Python with no dependencies. It aims to be convenient within
particular fields of science rather than to cover every unit in use.


Installation
============

::

    pip install pyqn

PyQn needs Python 3.10 or later.


Quick start
===========

>>> from pyqn import Units, Quantity
>>> Units("km.hr-1").conversion("m.s-1")
0.2777777777777778
>>> Units("J/mol.K")
J.mol-1.K-1
>>> str(Units("kJ.mol-1").dims)
'L2.M.T-2.Q-1'
>>> Units("J") == Units("kg.m2.s-2")
True
>>> Units("m.s-1") * Units("s")
m

Converting between units of different dimensions is an error, unless a
convention is asked for (see `Conventions`_):

>>> Units("J").conversion("K")
Traceback (most recent call last):
    ...
UnitsError: Failure in units conversion: units J[L2.M.T-2] and K[Theta] have different dimensions


Unit syntax
===========

A unit string is a sequence of *atoms* joined by ``.`` (multiply) or ``/``
(divide). Each atom is a unit symbol, perhaps with an SI prefix, followed by
an optional exponent:

==========  ==================================
``kg``      kilogram: prefix ``k``, unit ``g``
``s-1``     per second
``nm3``     cubic nanometre
``Pa-1/2``  a rational exponent
``m10``     metre to the power 10
==========  ==================================

- The exponent follows the unit directly, as an integer or a ratio of
  integers: not ``m^2`` or ``m**2``.
- Each ``/`` divides by everything after it up to the next ``/``:
  ``J/mol.K`` is J mol⁻¹ K⁻¹, and ``kg/m2/s`` is kg m⁻² s⁻¹.
- Spaces are not allowed: ``kg m-2`` is an error, not kg m⁻².
- ``1`` is the unit of a dimensionless quantity, as in ``1/s``.
- The SI prefixes are ``y z a f p n μ m c d k M G T P E Z Y``. Deca and
  hecto (``da``, ``h``) and the prefixes added in 2022 (ronna, quetta, ronto
  and quecto) are not supported.
- ``μ`` may be typed as the Greek letter mu or as the micro sign ``µ``; the
  ohm and angstrom signs are likewise read as ``Ω`` and ``Å``. A plain ``u``
  is the atomic mass unit, not micro.
- Symbols are case-sensitive: ``K`` is the kelvin and ``k`` the kayser.
- ``undef`` stands for undefined units. Products, ratios and powers
  involving ``undef`` are ``undef`` too; ``undef`` units cannot be converted,
  and quantities with them cannot be added or subtracted.

Every combination of prefix and unit symbol reads in only one way. To make
that possible, a few units have unusual symbols:

- ``inch``, not ``in``, because ``min`` would be both minutes and
  milli-inches;
- ``yard``, not ``yd`` (the yoctoday);
- ``dy`` for the day, because ``d`` is the prefix deci;
- ``asec`` for the arcsecond, because ``as`` is the attosecond.

Note also that ``b`` is the barn, so ``mb`` is the millibarn: the millibar
is ``mbar``.


Conventions
===========

``Units.conversion(other, force=...)`` can convert between units of
different dimensions using one of three conventions:

``force="spec"``
    Spectroscopy: energy, wavenumber and frequency are interconverted with
    E = hν = hcν̃.
``force="mol"``
    Molar quantities: the amount dimension is added or removed with the
    Avogadro constant, as between kJ.mol-1 and J (per molecule).
``force="kBT"``
    Energy and temperature are interconverted with E = k\ :sub:`B`\ T.

>>> Units("J").conversion("cm-1", force="spec")
5.0341165675427096e+22
>>> Units("eV").conversion("kJ.mol-1", force="mol")
96.48533212331002
>>> Units("K").conversion("J", force="kBT")
1.380649e-23

Only one convention can be used in each conversion.

Conversion factors use the exact values of the SI defining constants and the
CODATA 2022 values of the others (such as the electron mass and the
hartree).


Quantities
==========

A ``Quantity`` has a value, units, and optionally a name and a standard
uncertainty, ``sd``. Arithmetic on quantities checks their units and
propagates their uncertainties, assuming they are uncorrelated. An ``sd`` of
0 means the value is exact; an ``sd`` of ``None`` means its uncertainty is
unknown, and so is the uncertainty of anything calculated from it.

>>> q = Quantity.parse("x = 1.5(2)e-3 m.s-1")
>>> q.value, q.sd
(0.0015, 0.0002)
>>> q.as_str()
'x = 0.00150(20) m.s-1'
>>> t = Quantity(name="t", value=10.0, units="s", sd=0.5)
>>> (q * t).as_str()
'0.0150(21) m'
>>> q.convert_units_to("km.hr-1").as_str()
'0.00540(72) km.hr-1'

Quantities can be added or subtracted only if their units are equal: J and
kg.m2.s-2 can be added, but J and kJ cannot.


Limitations
===========

- Temperatures are absolute: there are no units with an offset, such as °C
  or °F.
- Angles are dimensionless, so, for example, ``Hz`` and ``rad.s-1`` are the
  same units.
- There are no logarithmic units, such as the decibel.


Units
=====

These are the units that PyQn recognises; each can take any SI prefix.

.. units-table-start

**Unity**

=====  =====  ===============
Unit   Name   Dimensions
=====  =====  ===============
``1``  unity  [dimensionless]
=====  =====  ===============

**SI unit stems**

=======  =======  ==========
Unit     Name     Dimensions
=======  =======  ==========
``m``    metre    L
``s``    second   T
``g``    gram     M
``K``    kelvin   Theta
``mol``  mole     Q
``A``    amp      C
``cd``   candela  I
=======  =======  ==========

**Derived SI units**

=======  =========  ===============
Unit     Name       Dimensions
=======  =========  ===============
``N``    newton     L.M.T-2
``J``    joule      L2.M.T-2
``W``    watt       L2.M.T-3
``Pa``   pascal     L-1.M.T-2
``C``    coulomb    T.C
``V``    volt       L2.M.T-3.C-1
``Ω``    ohm        L2.M.T-3.C-2
``F``    farad      L-2.M-1.T4.C2
``Wb``   weber      L2.M.T-2.C-1
``H``    henry      L2.M.T-2.C-2
``S``    siemens    L-2.M-1.T3.C2
``T``    tesla      M.T-2.C-1
``Hz``   hertz      T-1
``rad``  radian     [dimensionless]
``sr``   steradian  [dimensionless]
=======  =========  ===============

**Non-SI pressure units**

========  ======================  ==========
Unit      Name                    Dimensions
========  ======================  ==========
``bar``   bar                     L-1.M.T-2
``atm``   atmosphere              L-1.M.T-2
``Torr``  torr                    L-1.M.T-2
``mmHg``  millimetres of mercury  L-1.M.T-2
========  ======================  ==========

**cgs units**

========  ===============  ==========
Unit      Name             Dimensions
========  ===============  ==========
``dyn``   dyne             L.M.T-2
``erg``   erg              L2.M.T-2
``k``     kayser           L-1
``D``     debye            L.T.C
``hbar``  hbar             L2.M.T-1
``e``     electron charge  T.C
========  ===============  ==========

**Angular units**

==========  =========  ===============
Unit        Name       Dimensions
==========  =========  ===============
``deg``     degree     [dimensionless]
``arcmin``  arcminute  [dimensionless]
``asec``    arcsecond  [dimensionless]
==========  =========  ===============

**Non-SI energy units**

=======  ======================  ==========
Unit     Name                    Dimensions
=======  ======================  ==========
``eV``   electron volt           L2.M.T-2
``E_h``  hartree                 L2.M.T-2
``cal``  thermochemical calorie  L2.M.T-2
``Ry``   rydberg                 L2.M.T-2
=======  ======================  ==========

**Non-SI mass units**

=======  ================  ==========
Unit     Name              Dimensions
=======  ================  ==========
``u``    atomic mass unit  M
``amu``  atomic mass unit  M
``Da``   dalton            M
``m_e``  electron mass     M
=======  ================  ==========

**Non-SI units of length, area and volume**

======  ========  ==========
Unit    Name      Dimensions
======  ========  ==========
``Å``   angstrom  L
``a0``  bohr      L
``b``   barn      L2
``l``   litre     L3
``L``   litre     L3
======  ========  ==========

**Non-SI time units**

=======  ======  ==========
Unit     Name    Dimensions
=======  ======  ==========
``min``  minute  T
``hr``   hour    T
``h``    hour    T
``dy``   day     T
``yr``   year    T
=======  ======  ==========

**Astronomical units**

======  =================  ==========
Unit    Name               Dimensions
======  =================  ==========
``AU``  astronomical unit  L
``pc``  parsec             L
``ly``  light-year         L
======  =================  ==========

**Imperial, customary and US units**

=========  ====================  ==========
Unit       Name                  Dimensions
=========  ====================  ==========
``inch``   inch                  L
``ft``     foot                  L
``yard``   yard                  L
``fur``    furlong               L
``mi``     mile                  L
``gal``    Imperial (UK) gallon  L3
``pt``     Imperial (UK) pint    L3
``USgal``  US liquid gallon      L3
``USpt``   US liquid pint        L3
``st``     stone                 M
``lb``     pound                 M
``oz``     ounce                 M
=========  ====================  ==========

**Maritime units**

======  =============  ==========
Unit    Name           Dimensions
======  =============  ==========
``NM``  nautical mile  L
``kn``  knot           L.T-1
======  =============  ==========

**Silly units**

=======  =========  ==========
Unit     Name       Dimensions
=======  =========  ==========
``fir``  firkin     M
``ftn``  fortnight  T
=======  =========  ==========

**Miscellaneous units**

======  ========  ============
Unit    Name      Dimensions
======  ========  ============
``Td``  townsend  L4.M.T-3.C-1
``Jy``  jansky    M.T-2
======  ========  ============

.. units-table-end


Development
===========

Install the package in editable mode with its development tools (this needs
pip 25.1 or later)::

    python -m pip install -e . --group dev

Then run the tests, which include the examples in this README, and check the
formatting and the code::

    pytest
    ruff format --check .
    ruff check .

``python -m pyqn.list_base_units rst`` prints the table of units above, and
a test checks that the README has the current version.


Licence
=======

PyQn is free software, released under the GNU General Public License,
version 3 or later (see ``LICENSE``).
