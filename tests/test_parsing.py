"""Unit tests for parsing unit strings: syntax errors, Unicode and round trips."""

import unittest
import warnings

from pyqn.dimensions import Dimensions
from pyqn.units import Units, UnitsError


class ParsingCheck(unittest.TestCase):
    """Unit tests for parsing strings into Units objects."""

    def test_trailing_input_rejected(self):
        for s_units in ("kg m-2", "J mol-1", "m^2", "m**2", "m2s", "m-1x", "100"):
            with self.assertRaises(UnitsError, msg=s_units):
                Units(s_units)

    def test_exponents_containing_zero(self):
        self.assertEqual(Units("m10").dims, Dimensions(L=10))
        self.assertEqual(Units("s-20").dims, Dimensions(T=-20))
        self.assertEqual(str(Units("m10")), "m10")

    def test_bad_syntax_raises_units_error(self):
        for s_units in ("", "m..s", "/s", "m--1", "m-"):
            with self.assertRaises(Exception, msg=s_units) as cm:
                Units(s_units)
            self.assertIsInstance(cm.exception, UnitsError, msg=s_units)

    def test_unrecognised_unit(self):
        with self.assertRaises(UnitsError):
            Units("furlongs")

    def test_lookalike_unicode(self):
        # U+2126 OHM SIGN, U+00B5 MICRO SIGN and U+212B ANGSTROM SIGN should
        # be read as U+03A9, U+03BC and U+00C5, the characters in the unit table.
        self.assertEqual(str(Units("k\u2126")), "k\u03a9")
        self.assertEqual(str(Units("\u00b5s")), "\u03bcs")
        self.assertEqual(str(Units("\u212b")), "\u00c5")

    def test_round_trip(self):
        for s_units in (
            "m.s-1",
            "cm.hr-1/m",
            "kg1/2.m-3/2",
            "μs.J/m3",
            "kΩ",
            "mmHg",
            "Å-1",
            "a0-2",
            "E_h.m_e-1",
            "nmol2.L-1",
            "Pa-2/3.s4",
        ):
            u1 = Units(s_units)
            u2 = Units(str(u1))
            self.assertEqual(str(u2), str(u1), msg=s_units)
            self.assertEqual(u2, u1, msg=s_units)


class PrefixRulesCheck(unittest.TestCase):
    """Unit tests for the prefixes that each unit takes."""

    def test_allowed_prefixes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for s_units in (
                "kcal",
                "mbar",
                "mTorr",
                "MeV",
                "kDa",
                "Myr",
                "Mpc",
                "kly",
                "mD",
                "μu",
                "masec",
                "mE_h",
                "fb",
                "hPa",
                "hL",
                "hm",
                "hg",
            ):
                Units(s_units)
        self.assertEqual(Units("hPa"), Units("mbar"))
        self.assertEqual(str(Units("hbar")), "hbar")
        self.assertEqual(Units("hbar").dims, Dimensions(M=1, L=2, T=-1))

    def test_hecto_is_strict(self):
        # pyqn 2.0 had no hecto, so it is only accepted where the rules allow it
        for s_units in ("hJ", "hs", "hmin", "hft", "hyr"):
            with self.assertRaises(UnitsError, msg=s_units):
                Units(s_units)

    def test_deprecated_prefixes(self):
        # pyqn 2.0 took these; 2.1 warns, and 3.0 will reject them
        for s_units in ("kmin", "Mft", "katm", "μÅ", "mdy", "mu", "kD", "Gly"):
            with self.assertWarns(DeprecationWarning, msg=s_units) as cm:
                u1 = Units(s_units)
            self.assertEqual(str(u1), s_units)
            # the warning points to the caller, not to pyqn
            self.assertEqual(cm.filename, __file__)

    def test_new_si_units(self):
        self.assertEqual(Units("lm"), Units("cd.sr"))
        self.assertEqual(Units("lx"), Units("lm/m2"))
        self.assertEqual(Units("kBq").dims, Dimensions(T=-1))
        self.assertEqual(Units("mGy"), Units("mJ/kg"))
        self.assertEqual(Units("μSv").dims, Dimensions(L=2, T=-2))
        self.assertEqual(Units("nkat"), Units("nmol/s"))


if __name__ == "__main__":
    unittest.main()
