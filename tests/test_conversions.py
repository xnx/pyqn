# test_conversions.py
#
# Copyright (C) 2012-2016 Christian Hill
#
# Version 1.0
# Unit tests for unit conversions within the Units class.

import unittest
from pyqn.units import Units, UnitsError


class UnitsConversionCheck(unittest.TestCase):
    """Unit tests for unit conversions within the Units class."""

    def test_regular_units_conversion(self):
        u1 = Units("m.s-1")
        u2 = Units("cm.s-1")
        u3 = Units("ft.hr-1")
        u4 = Units("m.s-2")

        self.assertAlmostEqual(u1.conversion(u2), 100)
        self.assertAlmostEqual(u2.conversion(u1), 0.01)
        self.assertAlmostEqual(u1.conversion(u3), 11811.023622047243)
        with self.assertRaises(UnitsError) as cm:
            u1.conversion(u4)

    def test_litres(self):
        u1 = Units("l")
        self.assertEqual(u1.to_si(), 1.0e-3)
        u2 = Units("L")
        self.assertEqual(u1.to_si(), 1.0e-3)

    def test_molar_units_conversion(self):

        u1 = Units("mol")
        u2 = Units("1")
        self.assertAlmostEqual(u1.conversion(u2, force="mol"), 6.02214076e23)

        u1 = Units("mol-1")
        u2 = Units("1")
        self.assertAlmostEqual(u1.conversion(u2, force="mol"), 1 / 6.02214076e23)
        print(u1.dims.dims)

        u1 = Units("mol2")
        u2 = Units("1")
        self.assertAlmostEqual(u1.conversion(u2, force="mol"), 6.02214076e23**2)

        u1 = Units("mol-2")
        u2 = Units("1")
        self.assertAlmostEqual(u1.conversion(u2, force="mol"), 6.02214076e23**-2)

        u1 = Units("kJ")
        u2 = Units("kJ.mol-1")

        with self.assertRaises(UnitsError) as cm:
            u1.conversion(u2, strict=True)
        self.assertAlmostEqual(u1.conversion(u2, force="mol"), 1.660e-21)

        u3 = Units("cal.mol-1")
        u4 = Units("eV")
        self.assertAlmostEqual(u3.conversion(u4, force="mol"), 4.3364104350063916e-05)

        u9 = Units("nmol2.L-1")
        u10 = Units("L-1")
        self.assertAlmostEqual(u9.conversion(u10, force="mol"), 3.626617933325338e29)

        u5 = Units("J.mmol-1")
        u6 = Units("J")
        self.assertAlmostEqual(u5.conversion(u6, force="mol"), 1.6605390671738467e-21)

        u7 = Units("1")
        u8 = Units("mol-2")
        self.assertAlmostEqual(u7.conversion(u8, force="mol"), 2.757389993610589e-48)

        u9 = Units("nmol2.L-1")
        u10 = Units("L-1")
        self.assertAlmostEqual(u9.conversion(u10, force="mol"), 3.626617933325338e29)

    def test_kBT_units_conversion(self):
        u1 = Units("K")
        u2 = Units("J")

        self.assertAlmostEqual(u1.conversion(u2, force="kBT"), 1.38064852e-23)

    def test_spec_conversions(self):
        u1 = Units("J")
        u2 = Units("cm-1")
        u3 = Units("s-1")

        self.assertAlmostEqual(u1.conversion(u2, force="spec"), 5.0341165675427096e22)
        self.assertAlmostEqual(u1.conversion(u3, force="spec"), 1.5091901796421518e33)

    def test_rational_units_conversion(self):
        u1 = Units("Hz-1/2")
        u2 = Units("ns1/2")
        self.assertAlmostEqual(u1.conversion(u2), 10**4.5)

        u3 = Units("m-3/2")
        u4 = Units("inch-3/2")
        self.assertAlmostEqual(u3.conversion(u4), 0.0254**1.5)


if __name__ == "__main__":
    unittest.main()
