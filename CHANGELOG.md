# Changelog

## 2.1.0 (unreleased)

### Added

- The molecule, `molec`, a unit of amount (exactly 1/N_A mol), so that
  HITRAN's `cm-1/molec.cm-2` converts to `cm.mol-1`, or with `force="mol"`
  to `cm` per molecule.
- Pseudo-dimensions: named dimensions beyond the seven of the SI, which
  cancel only against themselves, for the kinds of fraction that the SI gives
  dimension 1. `ppmw`, `ppbw` and `wt%` are mass fractions; `ppmv`, `ppbv`,
  `pptv` and `vol%` volume fractions; `mol%` an amount fraction. Fractions of
  different kinds cannot be converted into each other, unless the conventions
  `force="mass fraction"`, `"volume fraction"` or `"amount fraction"` treat
  their pseudo-dimensions as dimensionless. `Dimensions` has a `pseudo`
  attribute.
- Conventions can be combined, as in
  `Units("kJ.mol-1").conversion("cm-1", force=("mol", "spec"))`. Each
  convention treats physical constants as 1 (`spec`: h and c; `kBT`: k_B;
  `mol`: N_A), and pyqn finds the powers of them by which the units'
  dimensions differ, so conventions apply to units of any dimensions, such as
  `cm-1.atm-1` and `MHz.Torr-1`. `convert()` takes `force` too.
- Domains: named sets of units, used when parsing unit strings, so that
  strings that make no sense in a field are rejected. `Units`,
  `Units.parse()`, `Quantity` and `Quantity.parse()` take a `domain`; the
  default, `domains.everything`, has every unit, and `domains.si` has the SI
  units and the non-SI units accepted for use with them. `Domain` makes
  domains of your own, checking that every string they accept reads in only
  one way, and `Domain.format()` writes units with a domain's spellings.
- Domains for particular fields, each with the units of `si` and those the
  field uses with them: `spectroscopy`, `atomic_and_molecular_physics`,
  `chemistry`, `astronomy`, `geology`, `gaussian_electromagnetism`,
  `radiation_physics` and `meteorology`. Some read a symbol in their field's
  own way: in geology `mD` is the millidarcy and `Ma` the mega-annum, and in
  radiation physics `rad` is the unit of absorbed dose.
- Units for these fields: the molar (`M`); `ppm`, `ppb`, `ppt` (parts per
  trillion) and `%`; the darcy, gal (`Gal`), `psi` and oil barrel (`bbl`);
  the Gaussian gauss (`G`), oersted (`Oe`), maxwell (`Mx`), statcoulomb
  (`statC`), statvolt (`statV`) and gamma (`γ`), defined by the SI units
  they correspond to; and the curie (`Ci`), röntgen (`R`), rem and rad
  (`rd`, so as not to be the radian).
- The standard symbols `in`, `yd` and `d` for the inch, yard and day. pyqn
  still prints these units as `inch`, `yard` and `dy`.
- The tonne (`t`) and the hectare (`ha`).
- The hecto prefix, `h`, for `Pa`, `L`, `l`, `m` and `g` (as in `hPa`).
- The SI derived units lumen (`lm`), lux (`lx`), becquerel (`Bq`), gray
  (`Gy`), sievert (`Sv`) and katal (`kat`).

### Changes

- Errors for conversions between units of different dimensions write the
  dimensions in parentheses, as in "units J (L2.M.T-2) and K (Theta)".
- Because `spec` treats c as 1, it also converts lengths to light-travel
  times (`cm` to `s`). An unknown convention name raises `UnitsError`.
- `ppt` is parts per trillion. pyqn 2.0 read it as the pico-pint.
- Errors for unrecognised units say why, as in "Unrecognised unit: hJ (the
  joule, J, does not take the prefix h)".
- Each unit takes only the SI prefixes that make sense for it: all of them
  for the SI units and units such as `eV`, `bar` and `cal`; a few for some
  (`kDa`, `Myr`, `Mpc`, `kly`, `mD`, `μu`); none for units such as `min`,
  `ft`, `atm` and `Å`. The README lists the prefixes for each unit. Prefixes
  that pyqn 2.0 accepted but a unit no longer takes, as in `kmin` or `μÅ`,
  still work but give a `DeprecationWarning`; pyqn 3.0 will reject them.
- `Units`, `Dimensions` and the objects they are built from (`AtomUnit`,
  `BaseUnit` and `SIPrefix`) are immutable: assigning to their attributes
  raises `dataclasses.FrozenInstanceError`, a kind of `AttributeError`.
  `Units.atom_units` is a tuple rather than a list.
- `Units` and `Dimensions` are hashable, so they can be used in sets and as
  dictionary keys; equal units, such as `J` and `kg.m2.s-2`, hash equal.
- Parsing units, and arithmetic with units and quantities, are about five
  times faster.

## 2.0.1 (25 October 2026)

### Fixes

- Arithmetic with `undef` units raised `AttributeError`. Products, ratios and
  powers involving `undef` are now `undef`; converting `undef` units, or
  adding or subtracting quantities with them, raises `UnitsError`.
- `q ** n` propagates the uncertainty of a `Quantity` (it was dropped).
- `Quantity.parse()` uses its `sd` and `definition` arguments (a standard
  deviation in the string takes precedence).

## 2.0.0 (25 October 2026)

### Breaking changes

- PyQn needs Python 3.10 or later, and no longer depends on pyparsing: it has
  no dependencies.
- Unit strings are parsed strictly. Strings that used to be misread silently
  now raise `UnitsError`: `"kg m-2"` (spaces are not allowed) was read as kg,
  `"m^2"` as m and `"100"` as 1.
- `Units` equality compares conversion factors with a relative tolerance
  (it used an absolute one, so that `pm == fm` and `eV == E_h`), and
  dimensionless units are equal only if their factors are: `deg != rad`, and
  `Units("deg") != 1`.
- `Units.conversion()` no longer takes the unused `strict` argument.
- The `pyqn.qn_array` module, which could not be imported, has been removed.

### Fixes

- Multiplying or dividing units containing the same unit (`cm * cm`,
  `"km.hr-1/hr"`) gave the wrong conversion factor, and multiplying could
  change the right-hand operand.
- Exponents containing 0, such as `m10`, are read correctly.
- The micro, ohm and angstrom signs are read as `μ`, `Ω` and `Å`: `"kΩ"`
  typed with the ohm sign was read as the kayser.
- Bad syntax raises `UnitsError` rather than pyparsing or `ValueError`
  exceptions.
- `force="mol"` conversions to molar units were wrong (`eV` to `kJ.mol-1`
  gave 2.7e-46), and conversions between units whose other dimensions differ
  were allowed.
- The knot was 3600 times too large.
- `1 / Units("s")` and `"m" / Units("s")` work; `u**0` has no units left;
  dimensions with fractional exponents print correctly.
- `Quantity`: `q * 2` and `2 * q` work; `q / 2` and `2 / q` give the right
  value and uncertainty; products and quotients with negative or zero values
  or exact (`sd=0`) operands propagate the uncertainty correctly;
  `as_str()` and `value_as_str()` show the right number of digits;
  `Quantity.parse()` no longer drops units that contain a space (it raises
  `UnitsError`).

### Other changes

- Constants use the exact values of the SI defining constants and CODATA
  2022 values of the others, so `e.V` and `eV` agree. The torr, mmHg,
  parsec, US gallon and US pint have their defined values.
- `Units`, `Quantity`, `Dimensions`, `convert` and the exceptions can be
  imported from `pyqn`, which also has `__version__`.
- `python -m pyqn.list_base_units text|html|rst` lists the units.
- The README documents the unit syntax and lists the units.
