import unittest
from ..quantity import Quantity, QuantityError

class QuantityManipulations(unittest.testCase):
    def test_quantity_init(self):
        
    def test_quantity_multiplication(self):
        q1 = Quantity(value=22.4,units='m/s')
        q2 = Quantity(value=2,units='s')
        
        q3 = q1*q2
        self.assertAlmostEqual(q3.value,44.8)
        self.asserEqual(q3.units,'m')
        
        #q4 = 2*q2 want to be able to do this
    
    def test_quantity_division(self):
        q1 = Quantity(value=39, units='J')
        q2 = Quantity(value=5, units='s')
        
        q3 = q1/q2
        self.assertAlmostEqual(q3.value,7.8)
        self.assertEqual(q3.units,'J.s-1')
    
    def test_quantity_addition(self):
    
    def test_quantity_subtraction(self):
        
if __name__ == '__main__':
    unittest.main()