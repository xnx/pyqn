import unittest
from ..base_unit import BaseUnit
from ..dimensions import d_length, Dimensions, d_voltage, d_current, d_time, d_mass

class BaseUnitCheck(unittest.TestCase):
    def test_base_unit_init(self):
        bu1 = BaseUnit('m', 'metre', 'length', 1., '', 'm', d_length)
        self.assertEqual(bu1.stem, 'm')
        self.assertEqual(bu1.name, 'metre')
        self.assertEqual(bu1.unit_type, 'length')
        self.assertEqual(bu1.fac, 1)
        self.assertEqual(bu1.description, '')
        self.assertEqual(bu1.latex, 'm')
        self.assertEqual(bu1.dims, Dimensions(L=1))
        
        bu2 = BaseUnit('Ω', 'ohm','electric resistance', 1, '',r'\Omega',d_voltage / d_current)
        self.assertEqual(bu2.stem, 'Ω')
        self.assertEqual(bu2.name, 'ohm')
        self.assertEqual(bu2.unit_type, 'electric resistance')
        self.assertEqual(bu2.fac, 1)
        self.assertEqual(bu2.description, '')
        self.assertEqual(bu2.latex, r'\Omega')
        self.assertEqual(bu2.dims, d_voltage/d_current)
        
    def test_base_unit_eq(self):
        bu1 = BaseUnit('m', 'metre', 'length', 1., '', 'm', d_length)
        bu2 = BaseUnit('s', 'second', 'time', 1., '', 's', d_time)
        bu3 = BaseUnit('g', 'gram', 'mass', 1.e-3, '', 'g', d_mass)
        bu4 = BaseUnit('K', 'kelvin', 'temperature', 1., '', 'K', Dimensions(Theta=1))
        self.assertFalse(bu1 == bu2)
        self.assertFalse(bu1 == bu3)
        self.assertFalse(bu1 == bu4)
        self.assertFalse(bu2 == bu3)
        self.assertFalse(bu2 == bu4)
        self.assertFalse(bu3 == bu4)
        
        bu5 = BaseUnit('k', '', '', 1, '', '', Dimensions(dims = []))
        self.assertFalse(bu4 == bu5)

if __name__ == '__main__':
    unittest.main()

