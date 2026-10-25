# test_conversions.py
#
# Copyright (C) 2012-2016 Christian Hill
#
# Version 1.0
# Unit tests for unit conversions within the Units class.

import unittest

import pytest

from pyqn.quantity import Quantity
from pyqn.units import Units, UnitsError, convert

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
        with self.assertRaises(UnitsError):
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
        with self.assertRaises(UnitsError):
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


class ComposedConventionsCheck(unittest.TestCase):
    """Unit tests for conversions with several conventions at once."""

    def test_composed_conventions(self):
        for from_units, to_units, force, fac in (
            ("kJ.mol-1", "cm-1", ("mol", "spec"), 1.0e3 / NA / (h * c * 100)),
            ("K", "cm-1", ("kBT", "spec"), kB / (h * c * 100)),
            ("MHz", "K", ("spec", "kBT"), h * 1.0e6 / kB),
            ("kcal.mol-1", "K", ("mol", "kBT"), 4184 / (NA * kB)),
            ("eV", "K", ("kBT",), e / kB),
        ):
            u1, u2 = Units(from_units), Units(to_units)
            self.assertEqual(u1.conversion(u2, force=force), approx(fac))
            # and back again, with the conventions in any order
            self.assertEqual(u2.conversion(u1, force=force[::-1]), approx(1 / fac))

    def test_force_types(self):
        fac = Units("kJ.mol-1").conversion("cm-1", force=("mol", "spec"))
        for force in (["spec", "mol"], {"mol", "spec"}, ("mol", "spec", "mol")):
            self.assertEqual(Units("kJ.mol-1").conversion("cm-1", force=force), fac)

    def test_other_dimensions(self):
        # conventions apply to any units whose dimensions differ by powers of
        # the constants, such as pressure-broadening coefficients
        self.assertEqual(
            Units("cm-1.atm-1").conversion("MHz.Torr-1", force="spec"),
            approx(c * 100 / 1.0e6 / 760),
        )
        # an entropy in units of R, and one in units of kB
        self.assertEqual(
            Units("J.K-1.mol-1").conversion("1", force=("mol", "kBT")),
            approx(1 / (NA * kB)),
        )
        self.assertEqual(Units("J.K-1").conversion("1", force="kBT"), approx(1 / kB))

    def test_light_travel_time(self):
        # spec treats c as 1, so it converts lengths to light-travel times
        self.assertEqual(Units("cm").conversion("s", force="spec"), approx(0.01 / c))
        self.assertEqual(Units("m.s-1").conversion("1", force="spec"), approx(1 / c))

    def test_convention_errors(self):
        for from_units, to_units, force in (
            ("cm", "MHz", "spec"),
            ("J", "m", ("mol", "spec", "kBT")),
            ("mol", "mol2", "mol"),
            ("mol.K", "J.mol-1", ("mol", "kBT")),
            ("J", "cm-1", None),
            ("J", "cm-1", ()),
            ("J", "cm-1", "spect"),
            ("J", "J", "spect"),
        ):
            with self.assertRaises(UnitsError, msg=(from_units, to_units, force)):
                Units(from_units).conversion(to_units, force=force)

    def test_conversion_methods(self):
        self.assertEqual(Units("J").spec_conversion("cm-1"), approx(1 / (100 * h * c)))
        self.assertEqual(Units("K").kBT_conversion("J"), approx(kB))
        self.assertEqual(Units("mol").mol_conversion("1"), approx(NA))
        self.assertEqual(
            convert("kJ.mol-1", "cm-1", force=("mol", "spec")),
            approx(1.0e3 / NA / (h * c * 100)),
        )
        q1 = Quantity(value=2.0, units="kJ.mol-1", sd=0.1)
        q2 = q1.convert_units_to("cm-1", force=("mol", "spec"))
        self.assertEqual(q2.value, approx(2.0e3 / NA / (h * c * 100)))
        self.assertEqual(q2.sd, approx(0.1e3 / NA / (h * c * 100)))


if __name__ == "__main__":
    unittest.main()
