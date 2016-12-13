# test_units.py
#
# Copyright (C) 2012-2016 Christian Hill
#
# Version 1.0
# Unit tests for the Units class.

import unittest
from ..units import Units
from ..dimensions import Dimensions,d_energy

class UnitsCheck(unittest.TestCase):
    """Unit tests for the Units class."""

    def test_units_algebra(self):
        u1 = Units('m.s-1')
        self.assertTrue(u1.has_units())
        u2 = Units('cm.hr-1/m')

        self.assertEqual(str(u1), 'm.s-1')
        self.assertEqual(str(u2), 'cm.hr-1.m-1')

        self.assertEqual(u1*u2, Units('cm.s-1.hr-1'))
        self.assertEqual(u1/u2, Units('m2.s-1.cm-1.hr'))
        self.assertEqual(u2/u1, Units('cm.hr-1.s/m2'))

        u3 = Units('J.A-1')
        u4 = Units('J.A-1')
        u3_over_u4 = u3 / u4
        self.assertFalse(u3_over_u4.has_units())

    def test_units_multiplication(self):
        u1 = Units('m.s-1')
        u2 = u1 * 1
        
        self.assertEqual(u2, u1)
        self.assertNotEqual(id(u1), id(u2)) 
        
        u2 = 1 * u1
        self.assertEqual(u2, u1)
        self.assertNotEqual(id(u1), id(u2)) 

        u1 = Units('m.s-1')
        u2 = Units('J')
        self.assertEqual('kg.m.s-1' * u1, u2)

        u1 = Units('m.s-1')
        u2 = Units('J')
        self.assertEqual(u1 * 'kg.m.s-1', u2)
        
        u1 = Units('mm2')
        u2 = Units('g/m2')
        u3 = u1 * u2
        self.assertAlmostEqual(u3.to_si(), 1.e-9)
        self.assertTupleEqual(u3.get_dims().dims,(0, 1, 0, 0, 0, 0, 0))
        
        u4 = Units('J/s')
        u5 = u4 * u2
        self.assertAlmostEqual(u5.to_si(), 1.e-3)
        self.assertEqual(u5.get_dims().dims,(0,2,-3,0,0,0,0))

    def test_units_division(self):
        u1 = Units('eV.mm')
        u2 = Units('K.cm')
        
        u3 = u1 / u2
        self.assertEqual(u3.get_dims().dims,(2,1,-2,-1,0,0,0))
        self.assertAlmostEqual(u3.to_si(),1.6*10**(-20))
        
        u4 = Units('s-1')
        u5 = u2 / ('eV'*u4)
        self.assertEqual(u5.get_dims(),Dimensions(T=1,Theta=1,L=1)/d_energy)      
    
    def test_units_power(self):
        u1 = Units('J.m')
        u2 = u1 ** 2
        self.assertEqual(u2.get_dims(),Dimensions(M=2,L=6,T=-4))
        
    def test_units_algebra_dimensions(self):
        u1 = Units('m')
        u2 = Units('m.s-1')
        u3 = u1 * u2
        u4 = Units('m2.s-1')
        self.assertEqual(u3.dims, u4.dims)

        u3 = u1 / u2
        u4 = Units('s')
        self.assertEqual(u3.dims, u4.dims)

    def test_unicode_units(self):
        u1 = Units('kΩ')
        self.assertEqual(str(u1), 'kΩ')

    def test_html(self):
        u1 = Units('m.s-1')
        self.assertEqual(u1.html, 'm s<sup>-1</sup>')
        u2 = Units('μs.J/m3')
        self.assertEqual(u2.html, 'μs J m<sup>-3</sup>')

if __name__ == '__main__':
    unittest.main()
