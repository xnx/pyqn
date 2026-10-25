"""Unit tests for domains: named sets of units used to parse unit strings."""

import dataclasses
import math
import unittest
import warnings

import pytest

from pyqn import Domain, Quantity, Units, domains
from pyqn.base_unit import base_unit_stems, metric
from pyqn.dimensions import Dimensions
from pyqn.units import UnitsError

c = 299792458.0


def approx(expected, rel=1.0e-9):
    """
    Compare with a relative tolerance only: pytest.approx's default absolute
    tolerance (1e-12) would accept any two very small factors.

    """
    return pytest.approx(expected, rel=rel, abs=0)


class EverythingCheck(unittest.TestCase):
    """Unit tests for the default domain, everything."""

    def test_standard_spellings(self):
        for spelling, stem in (("in", "inch"), ("yd", "yard"), ("d", "dy")):
            u1 = Units(spelling)
            self.assertEqual(u1, Units(stem))
            # pyqn prints the canonical symbol; format() uses the spelling
            self.assertEqual(str(u1), stem)
            self.assertEqual(domains.everything.format(u1), spelling)
        # min is still the minute, and cd the candela
        self.assertEqual(Units("min").to_si(), 60)
        self.assertEqual(Units("cd").dims, Dimensions(I=1))

    def test_spellings_are_strict(self):
        # 'in' is new in pyqn 2.1, so it takes no deprecated prefixes, but
        # 'inch' keeps those of pyqn 2.0 until 3.0
        with self.assertRaises(UnitsError):
            Units("kin")
        with self.assertWarns(DeprecationWarning):
            Units("kinch")

    def test_default_domain(self):
        self.assertEqual(Units("ft", domain=domains.everything), Units("ft"))


class SICheck(unittest.TestCase):
    """Unit tests for the SI domain."""

    def test_si_units(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for s_units in (
                "km.h-1",
                "kg.m-2",
                "hPa",
                "mL",
                "kDa",
                "MeV",
                "Mt",
                "ha",
                "d",
                "au",
                "lm",
                "mSv",
                "deg",
                "masec",
                "min",
            ):
                Units(s_units, domain=domains.si)
        self.assertEqual(Units("Mt", domain=domains.si), Units("Tg"))
        self.assertEqual(Units("ha", domain=domains.si), Units("hm2"))
        self.assertEqual(Units("au", domain=domains.si), Units("AU"))

    def test_canonical_symbols(self):
        # a unit's canonical symbol works in every domain that has the unit
        self.assertEqual(Units("dy", domain=domains.si), Units("d", domain=domains.si))
        self.assertEqual(Units("AU", domain=domains.si), Units("au", domain=domains.si))

    def test_not_si_units(self):
        for s_units in ("ft", "in", "hr", "atm", "kcal", "Å", "Torr", "k"):
            with self.assertRaises(UnitsError, msg=s_units) as cm:
                Units(s_units, domain=domains.si)
            self.assertIn("si domain", str(cm.exception))

    def test_no_deprecated_prefixes(self):
        # only the default domain keeps the prefixes that pyqn 2.0 allowed
        with self.assertRaises(UnitsError):
            Units("kmin", domain=domains.si)


class FormatCheck(unittest.TestCase):
    """Unit tests for writing units with a domain's spellings."""

    def test_format(self):
        u1 = Units("km.d-1", domain=domains.si)
        self.assertEqual(str(u1), "km.dy-1")
        self.assertEqual(domains.si.format(u1), "km.d-1")
        self.assertEqual(domains.si.format(Units("undef")), "undef")
        with self.assertRaises(UnitsError):
            domains.si.format(Units("ft"))

    def test_round_trip(self):
        for domain, s_units in (
            (domains.everything, "in.yd-1/d"),
            (domains.everything, "kg1/2.m-3/2"),
            (domains.si, "au.d-1"),
            (domains.si, "MeV.Da-1/2"),
        ):
            u1 = Units(s_units, domain=domain)
            self.assertEqual(Units(domain.format(u1), domain=domain), u1)
            self.assertEqual(Units(str(u1), domain=domain), u1)


class QuantityDomainCheck(unittest.TestCase):
    """Unit tests for parsing the units of quantities in a domain."""

    def test_quantity_domain(self):
        q1 = Quantity.parse("1.5(2) km.d-1", domain=domains.si)
        self.assertEqual(q1.units, Units("km.dy-1"))
        for make in (
            lambda: Quantity(value=1, units="ft", domain=domains.si),
            lambda: Quantity.parse("1.5 ft", domain=domains.si),
            lambda: Quantity.parse("1.5", units="ft", domain=domains.si),
        ):
            with self.assertRaises(UnitsError):
                make()


class FieldDomainsCheck(unittest.TestCase):
    """Unit tests for the built-in domains of particular fields."""

    def test_field_units(self):
        field_units = {
            domains.spectroscopy: (
                "cm-1",
                "MHz",
                "mD",
                "Å",
                "Torr",
                "kcal.mol-1",
                "cm-1/molec.cm-2",
            ),
            domains.atomic_and_molecular_physics: ("E_h", "a0-2", "m_e", "mb"),
            domains.chemistry: (
                "mM",
                "mol.L-1",
                "kDa",
                "mmHg",
                "ppm",
                "ppt",
                "%",
                "molec.cm-3",
                "wt%",
                "ppmv",
                "mol%",
            ),
            domains.astronomy: ("kpc", "Ga", "au", "Jy", "erg.s-1", "mG", "kly"),
            domains.geology: (
                "mD",
                "Ma",
                "Myr",
                "mGal",
                "kpsi",
                "bbl",
                "ft",
                "in",
                "wt%",
                "ppmw",
            ),
            domains.gaussian_electromagnetism: ("G", "kOe", "Mx", "Fr", "statV", "γ"),
            domains.radiation_physics: ("mrad", "μCi", "mR", "mrem", "mSv", "kBq"),
            domains.meteorology: (
                "hPa",
                "mbar",
                "kn",
                "NM",
                "ft",
                "ppb",
                "%",
                "ppbv",
                "pptv",
                "molec.cm-3",
            ),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for domain, strings in field_units.items():
                for s_units in strings:
                    u1 = Units(s_units, domain=domain)
                    self.assertEqual(Units(domain.format(u1), domain=domain), u1)
                    self.assertEqual(Units(str(u1), domain=domain), u1)

    def test_field_spellings(self):
        # the same string can mean different units in different fields
        geology_mD = Units("mD", domain=domains.geology)
        self.assertEqual(geology_mD, Units("mdarcy"))
        self.assertEqual(str(geology_mD), "mdarcy")
        self.assertEqual(Units("mD", domain=domains.spectroscopy), Units("mD"))
        self.assertNotEqual(geology_mD.dims, Units("mD").dims)
        self.assertEqual(Units("Ma", domain=domains.geology), Units("Myr"))
        self.assertEqual(Units("Ga", domain=domains.astronomy), Units("Gyr"))
        self.assertEqual(
            Units("Fr", domain=domains.gaussian_electromagnetism), Units("statC")
        )
        # in radiation physics, rad is the unit of absorbed dose
        rad = Units("rad", domain=domains.radiation_physics)
        self.assertEqual(rad, Units("rd"))
        self.assertEqual(rad, Units("cGy"))
        self.assertNotEqual(rad.dims, Units("rad").dims)
        self.assertEqual(Units("sr", domain=domains.radiation_physics), Units("sr"))

    def test_not_in_field(self):
        for domain, s_units in (
            (domains.si, "M"),
            (domains.si, "G"),
            (domains.si, "ppm"),
            (domains.chemistry, "ft"),
            (domains.spectroscopy, "darcy"),
            (domains.geology, "Å"),
            (domains.gaussian_electromagnetism, "T"),
            (domains.radiation_physics, "μrad.s-1"),
            (domains.si, "molec"),
            (domains.astronomy, "molec"),
            (domains.geology, "ppmv"),
            (domains.meteorology, "wt%"),
            (domains.geology, "mol%"),
        ):
            with self.assertRaises(UnitsError, msg=f"{s_units} in {domain.name}"):
                Units(s_units, domain=domain)

    def test_gaussian_conversions(self):
        gaussian = domains.gaussian_electromagnetism
        self.assertEqual(Units("G", domain=gaussian).conversion("T"), approx(1.0e-4))
        self.assertEqual(
            Units("Oe", domain=gaussian).conversion("A.m-1"),
            approx(1.0e3 / (4 * math.pi)),
        )
        self.assertEqual(
            Units("Fr", domain=gaussian).conversion("nC"), approx(0.1 / c * 1e9)
        )
        # the gauss and the oersted have different dimensions in SI
        with self.assertRaises(UnitsError):
            Units("G", domain=gaussian).conversion(Units("Oe", domain=gaussian))

    def test_ppt(self):
        # parts per trillion, not the pico-pint of pyqn 2.0
        self.assertEqual(Units("ppt").to_si(), 1.0e-12)
        self.assertEqual(
            Units("ppt", domain=domains.chemistry).conversion("ppb"), approx(1.0e-3)
        )

    def test_built_in_domains(self):
        for domain in domains.built_in_domains:
            self.assertIsInstance(domain, Domain)
            self.assertEqual(Units("1/s", domain=domain), Units("s-1"))


class DomainCheck(unittest.TestCase):
    """Unit tests for making domains."""

    def test_ambiguous_prefixed_unit(self):
        # an inch spelt 'in' that took prefixes would make 'min' ambiguous
        inch = dataclasses.replace(base_unit_stems["inch"], stem="in", prefixes=metric)
        with self.assertRaisesRegex(ValueError, "min is ambiguous"):
            Domain("test", units=[inch, base_unit_stems["min"]])

    def test_ambiguous_spelling(self):
        with self.assertRaisesRegex(ValueError, "D is ambiguous"):
            Domain(
                "test",
                units=[base_unit_stems["D"]],
                spellings={"D": base_unit_stems["m"]},
            )

    def test_invalid_symbols(self):
        for stem in ("", "x2", "a b", "m.s", "m/s", "x-"):
            with self.assertRaises(ValueError, msg=repr(stem)):
                Domain(
                    "test", units=[dataclasses.replace(base_unit_stems["m"], stem=stem)]
                )

    def test_domain_is_immutable(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            domains.si.name = "SI"
        with self.assertRaises(TypeError):
            domains.si.spellings["ft"] = base_unit_stems["ft"]

    def test_domain_of_your_own(self):
        lab = Domain("lab", units=[*domains.si.units, domains.everything.stems["Torr"]])
        self.assertEqual(Units("mTorr", domain=lab), Units("mTorr"))
        with self.assertRaises(UnitsError):
            Units("ft", domain=lab)


if __name__ == "__main__":
    unittest.main()
