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
- The SI prefixes are ``y z a f p n μ m c d h k M G T P E Z Y``, but each
  unit takes only some of them: see the *Prefixes* column in the table of
  `Units`_. Hecto (``h``) goes only with ``Pa``, ``L``, ``l``, ``m`` and
  ``g``, as in ``hPa``. Deca (``da``) and the prefixes added in 2022 (ronna,
  quetta, ronto and quecto) are not supported.
- pyqn 2.0 allowed every prefix but hecto on every unit. Prefixes that a unit
  no longer takes, as in ``kmin`` or ``μÅ``, still work in pyqn 2.1 but give
  a ``DeprecationWarning``; pyqn 3.0 will reject them.
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

These are the units that PyQn recognises, with the SI prefixes that each
takes ("all except h" is every SI prefix except hecto).

.. units-table-start

**Unity**

=====  =====  ===============  ========
Unit   Name   Dimensions       Prefixes
=====  =====  ===============  ========
``1``  unity  [dimensionless]  none
=====  =====  ===============  ========

**SI unit stems**

=======  =======  ==========  ============
Unit     Name     Dimensions  Prefixes
=======  =======  ==========  ============
``m``    metre    L           all
``s``    second   T           all except h
``g``    gram     M           all
``K``    kelvin   Theta       all except h
``mol``  mole     Q           all except h
``A``    amp      C           all except h
``cd``   candela  I           all except h
=======  =======  ==========  ============

**Derived SI units**

=======  =========  ===============  ============
Unit     Name       Dimensions       Prefixes
=======  =========  ===============  ============
``N``    newton     L.M.T-2          all except h
``J``    joule      L2.M.T-2         all except h
``W``    watt       L2.M.T-3         all except h
``Pa``   pascal     L-1.M.T-2        all
``C``    coulomb    T.C              all except h
``V``    volt       L2.M.T-3.C-1     all except h
``Ω``    ohm        L2.M.T-3.C-2     all except h
``F``    farad      L-2.M-1.T4.C2    all except h
``Wb``   weber      L2.M.T-2.C-1     all except h
``H``    henry      L2.M.T-2.C-2     all except h
``S``    siemens    L-2.M-1.T3.C2    all except h
``T``    tesla      M.T-2.C-1        all except h
``Hz``   hertz      T-1              all except h
``rad``  radian     [dimensionless]  all except h
``sr``   steradian  [dimensionless]  all except h
``lm``   lumen      I                all except h
``lx``   lux        L-2.I            all except h
``Bq``   becquerel  T-1              all except h
``Gy``   gray       L2.T-2           all except h
``Sv``   sievert    L2.T-2           all except h
``kat``  katal      T-1.Q            all except h
=======  =========  ===============  ============

**Non-SI pressure units**

========  ======================  ==========  ============
Unit      Name                    Dimensions  Prefixes
========  ======================  ==========  ============
``bar``   bar                     L-1.M.T-2   all except h
``atm``   atmosphere              L-1.M.T-2   none
``Torr``  torr                    L-1.M.T-2   all except h
``mmHg``  millimetres of mercury  L-1.M.T-2   none
========  ======================  ==========  ============

**cgs units**

========  ===============  ==========  ============
Unit      Name             Dimensions  Prefixes
========  ===============  ==========  ============
``dyn``   dyne             L.M.T-2     all except h
``erg``   erg              L2.M.T-2    all except h
``k``     kayser           L-1         none
``D``     debye            L.T.C       m
``hbar``  hbar             L2.M.T-1    none
``e``     electron charge  T.C         none
========  ===============  ==========  ============

**Angular units**

==========  =========  ===============  ========
Unit        Name       Dimensions       Prefixes
==========  =========  ===============  ========
``deg``     degree     [dimensionless]  none
``arcmin``  arcminute  [dimensionless]  none
``asec``    arcsecond  [dimensionless]  μ, m
==========  =========  ===============  ========

**Non-SI energy units**

=======  ======================  ==========  ============
Unit     Name                    Dimensions  Prefixes
=======  ======================  ==========  ============
``eV``   electron volt           L2.M.T-2    all except h
``E_h``  hartree                 L2.M.T-2    m
``cal``  thermochemical calorie  L2.M.T-2    all except h
``Ry``   rydberg                 L2.M.T-2    m
=======  ======================  ==========  ============

**Non-SI mass units**

=======  ================  ==========  ========
Unit     Name              Dimensions  Prefixes
=======  ================  ==========  ========
``u``    atomic mass unit  M           μ
``amu``  atomic mass unit  M           none
``Da``   dalton            M           k, M
``m_e``  electron mass     M           none
=======  ================  ==========  ========

**Non-SI units of length, area and volume**

======  ========  ==========  ============
Unit    Name      Dimensions  Prefixes
======  ========  ==========  ============
``Å``   angstrom  L           none
``a0``  bohr      L           none
``b``   barn      L2          all except h
``l``   litre     L3          all
``L``   litre     L3          all
======  ========  ==========  ============

**Non-SI time units**

=======  ======  ==========  ========
Unit     Name    Dimensions  Prefixes
=======  ======  ==========  ========
``min``  minute  T           none
``hr``   hour    T           none
``h``    hour    T           none
``dy``   day     T           none
``yr``   year    T           k, M, G
=======  ======  ==========  ========

**Astronomical units**

======  =================  ==========  ========
Unit    Name               Dimensions  Prefixes
======  =================  ==========  ========
``AU``  astronomical unit  L           none
``pc``  parsec             L           k, M, G
``ly``  light-year         L           k
======  =================  ==========  ========

**Imperial, customary and US units**

=========  ====================  ==========  ========
Unit       Name                  Dimensions  Prefixes
=========  ====================  ==========  ========
``inch``   inch                  L           none
``ft``     foot                  L           none
``yard``   yard                  L           none
``fur``    furlong               L           none
``mi``     mile                  L           none
``gal``    Imperial (UK) gallon  L3          none
``pt``     Imperial (UK) pint    L3          none
``USgal``  US liquid gallon      L3          none
``USpt``   US liquid pint        L3          none
``st``     stone                 M           none
``lb``     pound                 M           none
``oz``     ounce                 M           none
=========  ====================  ==========  ========

**Maritime units**

======  =============  ==========  ========
Unit    Name           Dimensions  Prefixes
======  =============  ==========  ========
``NM``  nautical mile  L           none
``kn``  knot           L.T-1       none
======  =============  ==========  ========

**Silly units**

=======  =========  ==========  ========
Unit     Name       Dimensions  Prefixes
=======  =========  ==========  ========
``fir``  firkin     M           none
``ftn``  fortnight  T           none
=======  =========  ==========  ========

**Miscellaneous units**

======  ========  ============  ============
Unit    Name      Dimensions    Prefixes
======  ========  ============  ============
``Td``  townsend  L4.M.T-3.C-1  none
``Jy``  jansky    M.T-2         all except h
======  ========  ============  ============

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
