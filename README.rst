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

The documentation, with worked examples for spectroscopy, atmospheric
chemistry, physical chemistry and the validation of data, is at
https://pyqn.readthedocs.io.


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
UnitsError: Failure in units conversion: units J (L2.M.T-2) and K (Theta) have different dimensions


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

Every string reads in only one way. Because each unit takes only some
prefixes, the standard symbols ``in`` (inch), ``yd`` (yard) and ``d`` (day)
are safe to use: ``min`` is still the minute and ``cd`` the candela. pyqn
prints these units with the symbols it used before version 2.1, ``inch``,
``yard`` and ``dy``, which every version reads (see `Domains`_). The
arcsecond is ``asec``, because ``as`` is the attosecond.

Note also that ``b`` is the barn, so ``mb`` is the millibarn: the millibar
is ``mbar``. And ``ppt`` is always parts per trillion, never parts per
thousand.


Domains
=======

A *domain* is a named set of units, used when parsing unit strings, so that
strings that make no sense in a field are rejected. The default domain,
``domains.everything``, has every unit.

>>> from pyqn import domains
>>> Units("km.h-1", domain=domains.si)
km.h-1
>>> Units("ft", domain=domains.si)
Traceback (most recent call last):
    ...
UnitsError: Unrecognised unit in the si domain: ft (the tonne, t, does not take the prefix f)

``domains.si`` has the SI units and the non-SI units accepted for use with
them: the minute, hour, day, astronomical unit, degree, arcminute,
arcsecond, hectare, litre, tonne, dalton and electronvolt.

Each unit has a *canonical symbol*, which means the same in every domain and
is what pyqn prints, so what pyqn prints can be read back in any domain that
has the unit. A domain can also accept other spellings of a unit, and its
``format()`` writes units with them:

>>> u = Units("km.d-1", domain=domains.si)
>>> u
km.dy-1
>>> domains.si.format(u)
'km.d-1'

Once parsed, units are the same whatever domain they came from: arithmetic,
comparisons and conversions work in the same way, and a string in them, as
in ``u.conversion("m.s-1")``, is read in the default domain. ``Quantity`` and
``Quantity.parse()`` also take a ``domain``.

You can make domains of your own. When a domain is made, pyqn checks that
every string it accepts reads in only one way:

>>> from pyqn import Domain
>>> lab = Domain("lab", units=[*domains.si.units, domains.everything.stems["Torr"]])
>>> Units("mTorr", domain=lab)
mTorr


Built-in domains
----------------

pyqn has domains for several fields. Each has the units of ``si`` and those
that the field uses with them, and some read a symbol in the field's own
way. In geology, ``D`` is the darcy, so ``mD`` is the millidarcy, which pyqn
prints with the darcy's canonical symbol; in radiation physics, ``rad`` is
the unit of absorbed dose, and there is no radian.

>>> Units("mD", domain=domains.geology)
mdarcy
>>> domains.geology.format(Units("mD", domain=domains.geology))
'mD'
>>> Units("mD", domain=domains.spectroscopy)
mD

Gaussian (cgs) units belong to a system with three base dimensions, so pyqn
defines them by the SI units they correspond to. They convert to those
units, but the gauss and the oersted, for example, cannot be converted into
each other, because the tesla and the ampere per metre have different
dimensions:

>>> Units("G", domain=domains.gaussian_electromagnetism).conversion("T")
0.0001

.. domains-table-start

========================================  ========================================  =====
Domain                                    Other spellings                           Units
========================================  ========================================  =====
``domains.everything``                    ``in`` (inch), ``yd`` (yard), ``d`` (dy)  every unit (see `Units`_)
``domains.si``                            ``d`` (dy), ``au`` (AU)                   the SI units, and ``min``, ``h``, ``deg``, ``arcmin``, ``asec``, ``ha``, ``L``, ``l``, ``t``, ``Da``, ``u``, ``eV``, ``dy``, ``AU``
``domains.spectroscopy``                  none                                      those of ``si``, and ``Å``, ``D``, ``k``, ``Torr``, ``atm``, ``bar``, ``erg``, ``cal``, ``molec``
``domains.atomic_and_molecular_physics``  none                                      those of ``si``, and ``Å``, ``D``, ``E_h``, ``Ry``, ``a0``, ``m_e``, ``e``, ``hbar``, ``amu``, ``b``, ``molec``
``domains.chemistry``                     none                                      those of ``si``, and ``M``, ``Å``, ``D``, ``cal``, ``atm``, ``bar``, ``Torr``, ``mmHg``, ``molec``, ``ppm``, ``ppb``, ``ppt``, ``%``, ``ppmw``, ``ppbw``, ``wt%``, ``ppmv``, ``ppbv``, ``pptv``, ``vol%``, ``mol%``
``domains.astronomy``                     ``a`` (yr)                                those of ``si``, and ``pc``, ``ly``, ``yr``, ``Jy``, ``erg``, ``dyn``, ``Å``, ``G``
``domains.geology``                       ``a`` (yr), ``D`` (darcy), ``in`` (inch)  those of ``si``, and ``yr``, ``darcy``, ``Gal``, ``psi``, ``bbl``, ``bar``, ``atm``, ``ft``, ``inch``, ``ppm``, ``ppb``, ``ppt``, ``%``, ``ppmw``, ``ppbw``, ``wt%``
``domains.gaussian_electromagnetism``     ``Fr`` (statC)                            ``1``, ``m``, ``g``, ``s``, ``dyn``, ``erg``, ``G``, ``Oe``, ``Mx``, ``statC``, ``statV``, ``γ``
``domains.radiation_physics``             ``rad`` (rd)                              those of ``si`` except ``rad``, and ``Ci``, ``R``, ``rem``, ``rd``
``domains.meteorology``                   none                                      those of ``si``, and ``bar``, ``atm``, ``kn``, ``NM``, ``ft``, ``%``, ``ppm``, ``ppb``, ``ppt``, ``molec``, ``ppmv``, ``ppbv``, ``pptv``, ``vol%``
========================================  ========================================  =====

.. domains-table-end


Conventions
===========

``Units.conversion(other, force=...)`` can convert between units of
different dimensions using conventions, each of which treats some physical
constants as 1:

``force="spec"``
    Spectroscopy, with h and c: energy, wavenumber and frequency are
    interconverted with E = hν = hcν̃.
``force="mol"``
    Molar quantities, with the Avogadro constant: the amount dimension is
    added or removed, as between kJ.mol-1 and J (per molecule).
``force="kBT"``
    Energy and temperature, with the Boltzmann constant: E = k\ :sub:`B`\ T.

>>> Units("J").conversion("cm-1", force="spec")
5.0341165675427096e+22
>>> Units("eV").conversion("kJ.mol-1", force="mol")
96.48533212331002
>>> Units("K").conversion("J", force="kBT")
1.380649e-23

Conventions can be combined:

>>> Units("kJ.mol-1").conversion("cm-1", force=("mol", "spec"))
83.59347229111778
>>> Units("K").conversion("cm-1", force=("kBT", "spec"))
0.6950348004861274

pyqn finds the powers of the constants by which the dimensions of the two
units differ. The constants have independent dimensions, so there is at most
one way to use them, and a conversion that needs any other change of
dimensions is an error. The units can have any dimensions, so pressure
broadening coefficients, for example, convert too:

>>> Units("cm-1.atm-1").conversion("MHz.Torr-1", force="spec")
39.44637605263158

Two consequences: because ``spec`` treats c as 1, it also converts lengths to
light-travel times (``cm`` to ``s``); and ``mol`` only adds or removes the
amount dimension entirely, so ``mol`` cannot be converted to ``mol2``.
Wavelengths are not covered, because a wavelength is inversely proportional
to frequency rather than a multiple of it.

Conversion factors use the exact values of the SI defining constants and the
CODATA 2022 values of the others (such as the electron mass and the
hartree).


Molecules and fractions
=======================

A molecule, ``molec``, is a unit of amount: exactly 1/N\ :sub:`A` mol. So
HITRAN's line intensities, in cm\ :sup:`-1`/(molecule cm\ :sup:`-2`),
convert to molar units directly, and to units per molecule with
``force="mol"``:

>>> s = Units("cm-1/molec.cm-2")
>>> s
cm.molec-1
>>> s.conversion("cm.mol-1")
6.02214076e+23
>>> s.conversion("cm", force="mol")
1.0

Mass, volume and amount fractions all have dimension 1 in the SI, but cannot
be converted into each other without molar masses or densities. So the units
for a particular kind of fraction have a *pseudo-dimension*, which cancels
only against itself:

================  ========================================
Pseudo-dimension  Units
================  ========================================
mass fraction     ``ppmw``, ``ppbw``, ``wt%``
volume fraction   ``ppmv``, ``ppbv``, ``pptv``, ``vol%``
amount fraction   ``mol%``
================  ========================================

>>> Units("wt%").conversion("ppmw")
10000.0
>>> Units("ppmv").conversion("ppmw")
Traceback (most recent call last):
    ...
UnitsError: Failure in units conversion: units ppmv ([volume fraction]) and ppmw ([mass fraction]) have different dimensions

The plain ``ppm``, ``ppb``, ``ppt`` and ``%`` have dimension 1, and can stand
for any kind of fraction. A convention named after a pseudo-dimension treats
it as dimensionless:

>>> Units("ppmv").conversion("μmol/mol", force="volume fraction")
1.0


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

These are the units in the default domain, with their canonical symbols
and the SI prefixes that each takes ("all except h" is every SI prefix except
hecto). The default domain also reads ``in``, ``yd`` and ``d`` as the inch,
yard and day.

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
``t``    tonne             M           k, M, G
``m_e``  electron mass     M           none
=======  ================  ==========  ========

**Non-SI units of length, area and volume**

======  ========  ==========  ============
Unit    Name      Dimensions  Prefixes
======  ========  ==========  ============
``Å``   angstrom  L           none
``a0``  bohr      L           none
``b``   barn      L2          all except h
``ha``  hectare   L2          none
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

**Amounts, concentrations and fractions**

=========  ============================  =================  =============
Unit       Name                          Dimensions         Prefixes
=========  ============================  =================  =============
``molec``  molecule                      Q                  none
``M``      molar                         L-3.Q              f, p, n, μ, m
``ppm``    parts per million             [dimensionless]    none
``ppb``    parts per billion             [dimensionless]    none
``ppt``    parts per trillion            [dimensionless]    none
``%``      percent                       [dimensionless]    none
``ppmw``   parts per million by mass     [mass fraction]    none
``ppbw``   parts per billion by mass     [mass fraction]    none
``wt%``    percent by mass               [mass fraction]    none
``ppmv``   parts per million by volume   [volume fraction]  none
``ppbv``   parts per billion by volume   [volume fraction]  none
``pptv``   parts per trillion by volume  [volume fraction]  none
``vol%``   percent by volume             [volume fraction]  none
``mol%``   percent by amount             [amount fraction]  none
=========  ============================  =================  =============

**Geological and petroleum units**

=========  =====================  ==========  ========
Unit       Name                   Dimensions  Prefixes
=========  =====================  ==========  ========
``darcy``  darcy                  L2          m
``Gal``    gal (galileo)          L.T-2       μ, m
``psi``    pound per square inch  L-1.M.T-2   k
``bbl``    barrel (oil)           L3          none
=========  =====================  ==========  ========

**Gaussian electromagnetic units**

=========  ===========  ============  ========
Unit       Name         Dimensions    Prefixes
=========  ===========  ============  ========
``G``      gauss        M.T-2.C-1     μ, m, k
``Oe``     oersted      L-1.C         k
``Mx``     maxwell      L2.M.T-2.C-1  none
``statC``  statcoulomb  T.C           none
``statV``  statvolt     L2.M.T-3.C-1  none
``γ``      gamma        M.T-2.C-1     none
=========  ===========  ============  ========

**Radiation units**

=======  ===================  ==========  =============
Unit     Name                 Dimensions  Prefixes
=======  ===================  ==========  =============
``Ci``   curie                T-1         p, n, μ, m, k
``R``    röntgen              M-1.T.C     μ, m
``rem``  rem                  L2.T-2      m
``rd``   rad (absorbed dose)  L2.T-2      m
=======  ===================  ==========  =============

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

The documentation, in ``docs/``, includes this README as its user guide. To
build it::

    python -m pip install -e . --group docs
    sphinx-build -W -b html docs docs/_build/html


Licence
=======

PyQn is free software, released under the GNU General Public License,
version 3 or later (see ``LICENSE``).
