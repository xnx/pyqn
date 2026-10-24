# test_conversions.py
#
# Copyright (C) 2012-2016 Christian Hill
#
# Version 1.0
# Unit tests for unit conversions within the Units class.

import unittest
import pytest
from pyqn.units import Units, UnitsError

# Exact values of the SI defining constants.
h, c, kB, NA = 6.62607015e-34, 299792458.0, 1.380649e-23, 6.02214076e23
e = 1.602176634e-19


def approx(expected, rel=1.0e-6):
    """
    Compare with a relative tolerance only: pytest.approx's default absolute
    tolerance (1e-12) would accept any two very small conversion factors.

    """
    return pytest.approx(expected, rel=rel, abs=0)


class UnitsConversionCheck(unittest.TestCase):
    """Unit tests for unit conversions within the Units class."""

    def test_regular_units_conversion(self):
        u1 = Units("m.s-1")
        u2 = Units("cm.s-1")
        u3 = Units("ft.hr-1")
        u4 = Units("m.s-2")

        self.assertEqual(u1.conversion(u2), approx(100))
        self.assertEqual(u2.conversion(u1), approx(0.01))
        self.assertEqual(u1.conversion(u3), approx(3600 / 0.3048))
        with self.assertRaises(UnitsError) as cm:
            u1.conversion(u4)

    def test_litres(self):
        u1 = Units("l")
        self.assertEqual(u1.to_si(), 1.0e-3)
        u2 = Units("L")
        self.assertEqual(u2.to_si(), 1.0e-3)

    def test_molar_units_conversion(self):
        u1 = Units("mol")
        u2 = Units("1")
        self.assertEqual(u1.conversion(u2, force="mol"), approx(NA))

        u1 = Units("mol-1")
        self.assertEqual(u1.conversion(u2, force="mol"), approx(1 / NA))

        u1 = Units("mol2")
        self.assertEqual(u1.conversion(u2, force="mol"), approx(NA**2))

        u1 = Units("mol-2")
        self.assertEqual(u1.conversion(u2, force="mol"), approx(NA**-2))

        u1 = Units("kJ")
        u2 = Units("kJ.mol-1")
        with self.assertRaises(UnitsError) as cm:
            u1.conversion(u2)

        u3 = Units("cal.mol-1")
        u4 = Units("eV")
        self.assertEqual(u3.conversion(u4, force="mol"), approx(4.184 / NA / e))

        u5 = Units("J.mmol-1")
        u6 = Units("J")
        self.assertEqual(u5.conversion(u6, force="mol"), approx(1000 / NA))

        u9 = Units("nmol2.L-1")
        u10 = Units("L-1")
        self.assertEqual(u9.conversion(u10, force="mol"), approx((1.0e-9 * NA) ** 2))

    def test_to_molar_units_conversion(self):
        u1 = Units("kJ")
        u2 = Units("kJ.mol-1")
        self.assertEqual(u1.conversion(u2, force="mol"), approx(NA))

        u3 = Units("eV")
        self.assertEqual(u3.conversion(u2, force="mol"), approx(e * NA / 1000))

        u4 = Units("1")
        self.assertEqual(u4.conversion("mol", force="mol"), approx(1 / NA))
        self.assertEqual(u4.conversion("mol-2", force="mol"), approx(NA**2))

    def test_molar_conversion_checks_other_dimensions(self):
        for from_units, to_units in (("J", "K"), ("J.mol-1", "K")):
            with self.assertRaises(UnitsError, msg=f"{from_units} -> {to_units}"):
                Units(from_units).conversion(to_units, force="mol")

    def test_kBT_units_conversion(self):
        u1 = Units("K")
        u2 = Units("J")

        self.assertEqual(u1.conversion(u2, force="kBT"), approx(kB))
        self.assertEqual(u2.conversion(u1, force="kBT"), approx(1 / kB))

    def test_spec_conversions(self):
        u1 = Units("J")
        u2 = Units("cm-1")
        u3 = Units("s-1")

        self.assertEqual(u1.conversion(u2, force="spec"), approx(1 / (100 * h * c)))
        self.assertEqual(u1.conversion(u3, force="spec"), approx(1 / h))
        self.assertEqual(u2.conversion(u3, force="spec"), approx(100 * c))

    def test_rational_units_conversion(self):
        u1 = Units("Hz-1/2")
        u2 = Units("ns1/2")
        self.assertEqual(u1.conversion(u2), approx(10**4.5))

        u3 = Units("m-3/2")
        u4 = Units("inch-3/2")
        self.assertEqual(u3.conversion(u4), approx(0.0254**1.5))


if __name__ == "__main__":
    unittest.main()
