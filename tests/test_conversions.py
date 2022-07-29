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
        u1 = Units('m.s-1')
        u2 = Units('cm.s-1')
        u3 = Units('ft.hr-1')
        u4 = Units('m.s-2')

        self.assertAlmostEqual(u1.conversion(u2), 100)
        self.assertAlmostEqual(u2.conversion(u1), 0.01)
        self.assertAlmostEqual(u1.conversion(u3), 11811.023622047243)
        with self.assertRaises(UnitsError) as cm:
            u1.conversion(u4)

    def test_litres(self):
        u1 = Units('l')
        self.assertEqual(u1.to_si(), 1.e-3)
        u2 = Units('L')
        self.assertEqual(u1.to_si(), 1.e-3)

    def test_molar_units_conversion(self):
        u1 = Units('kJ')
        u2 = Units('J/mol')

        with self.assertRaises(UnitsError) as cm:
            u1.conversion(u2, strict=True)
        self.assertAlmostEqual(u1.conversion(u2, force='mol'), 1.660e-21)
    
    def test_kBT_units_conversion(self):
        u1 = Units('K')
        u2 = Units('J')
        
        self.assertAlmostEqual(u1.conversion(u2, force='kBT'), 1.38064852e-23)
    
    def test_spec_conversions(self):
        u1 = Units('J')
        u2 = Units('cm-1')
        u3 = Units('s-1')
        
        self.assertAlmostEqual(u1.conversion(u2, force='spec'),
                                                    5.0341165675427096e+22)
        self.assertAlmostEqual(u1.conversion(u3, force='spec'),
                                                    1.5091901796421518e+33)
 
if __name__ == '__main__':
    unittest.main()

