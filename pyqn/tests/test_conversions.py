# test_conversions.py
#
# Copyright (C) 2012-2016 Christian Hill
#
# Version 1.0
# Unit tests for unit conversions within the Units class.

import unittest
from ..units import Units, UnitsError

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

    def test_molar_units_conversion(self):
        u1 = Units('kJ')
        u2 = Units('J/mol')

        with self.assertRaises(UnitsError) as cm:
            u1.conversion(u2, strict=True)
        self.assertAlmostEqual(u1.conversion(u2,force='mol'), 1.6605389209999998e-21)
    
    def test_kbt_units_conversion(self):
        u1 = Units('K')
        u2 = Units('J')
        
        self.assertAlmostEqual(u1.conversion(u2,force='kbt'),1.38064852e-23)

    def test_multiplication_conversion(self):
        u1 = Units('mm2')
        u2 = Units('g/m2')
        u3 = u1 * u2
        self.assertAlmostEqual(u3.to_si(), 1.e-9)
        self.assertEqual(u3.dims()=='M')
    
    def test_division_conversion(self):
        u1 = Units('eV.mm')
        u2 = Units('K.cm')
        u3 = u1 / u2
        
        %self.assertAlmostEqual(u3.to_si(),)
        

if __name__ == '__main__':
    unittest.main()

