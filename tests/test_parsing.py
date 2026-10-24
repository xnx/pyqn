"""Unit tests for parsing unit strings: syntax errors, Unicode and round trips."""

import unittest
import pytest
from pyqn.units import Units, UnitsError
from pyqn.dimensions import Dimensions


class ParsingCheck(unittest.TestCase):
    """Unit tests for parsing strings into Units objects."""

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="parseString is called without parseAll=True",
    )
    def test_trailing_input_rejected(self):
        for s_units in ("kg m-2", "J mol-1", "m^2", "m**2", "m2s", "m-1x", "100"):
            with self.assertRaises(UnitsError, msg=s_units):
                Units(s_units)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="the exponent characters don't include 0",
    )
    def test_exponents_containing_zero(self):
        self.assertEqual(Units("m10").dims, Dimensions(L=10))
        self.assertEqual(Units("s-20").dims, Dimensions(T=-20))
        self.assertEqual(str(Units("m10")), "m10")

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="bad syntax raises ParseException or ValueError",
    )
    def test_bad_syntax_raises_units_error(self):
        for s_units in ("", "m..s", "/s", "m--1", "m-"):
            with self.assertRaises(Exception, msg=s_units) as cm:
                Units(s_units)
            self.assertIsInstance(cm.exception, UnitsError, msg=s_units)

    def test_unrecognised_unit(self):
        with self.assertRaises(UnitsError):
            Units("furlongs")

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="unit strings are not NFKC-normalised",
    )
    def test_lookalike_unicode(self):
        # U+2126 OHM SIGN, U+00B5 MICRO SIGN and U+212B ANGSTROM SIGN should
        # be read as U+03A9, U+03BC and U+00C5, the characters in the unit table.
        self.assertEqual(str(Units("kΩ")), "kΩ")
        self.assertEqual(str(Units("µs")), "μs")
        self.assertEqual(str(Units("Å")), "Å")

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


if __name__ == "__main__":
    unittest.main()
