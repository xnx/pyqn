import unittest
from ..quantity import Quantity, QuantityError
from ..dimensions import Dimensions, d_energy

class QuantityManipulations(unittest.TestCase):
    def test_quantity_init(self):
        pass
        
    def test_quantity_multiplication(self):
        q1 = Quantity(value=22.4,units='m/s')
        q2 = Quantity(value=2,units='s')
        
        q3 = q1*q2
        self.assertAlmostEqual(q3.value,44.8)
        self.assertEqual(q3.units.dims.dims,(1,0,0,0,0,0,0))
        
        #q4 = 2*q2 want to be able to do this
    
    def test_quantity_division(self):
        q1 = Quantity(value=39, units='J')
        q2 = Quantity(value=5, units='s')
        
        q3 = q1/q2
        self.assertAlmostEqual(q3.value,7.8)
        self.assertEqual(q3.units.dims,Dimensions(M=1,L=2,T=-3))
        q3 = q2/q1
        self.assertAlmostEqual(q3.value,0.128)
        self.assertEqual(q3.units.dims,Dimensions(M=-1,L=-2,T=3))
        
    
    def test_quantity_addition(self):
        q1 = Quantity(value = 20.5, units = 'J')
        q2 = Quantity(value = 30.7, units = 'kg.m2.s-2')
        q3 = Quantity(value = 5.1, units = 'K')
        
        q4 = q1+q2
        self.assertEqual(q4.value,51.2)
        self.assertEqual(q4.units.dims,Dimensions(M=1,L=2,T=-2))
        
        self.assertRaises(UnitsError, Quantity, q1+q3)
        
    def test_quantity_subtraction(self):
        pass
        
if __name__ == '__main__':
    unittest.main()