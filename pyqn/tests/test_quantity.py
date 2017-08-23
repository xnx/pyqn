import unittest
from ..quantity import Quantity, QuantityError
from ..dimensions import Dimensions, d_energy
from ..units import UnitsError, Units

class QuantityManipulations(unittest.TestCase):
    def test_quantity_init(self):
        pass
        
    def test_quantity_parse(self):
        q1 = Quantity.parse("a = 10 m/s")
        q2 = Quantity.parse("lambda = 300.15(10) nm")
        q3 = Quantity.parse("1e5 J")
        
        self.assertEqual(q1.name, 'a')
        self.assertEqual(q1.value, 10)
        self.assertEqual(q1.units, Units('m.s-1'))
        
        self.assertEqual(q2.name, 'lambda')
        self.assertEqual(q2.value, 300.15)
        self.assertEqual(q2.sd, 0.1)
        self.assertEqual(q2.units, Units('nm'))
        
        self.assertEqual(q3.value, 1e5)
        self.assertEqual(q3.units, Units('J'))
        
    def test_quantity_multiplication(self):
        q1 = Quantity(value=22.4,units='m/s')
        q2 = Quantity(value=2,units='s')
        
        q3 = q1*q2
        self.assertAlmostEqual(q3.value,44.8)
        self.assertEqual(q3.units.dims.dims,(1,0,0,0,0,0,0))
    
    def test_quantity_division(self):
        q1 = Quantity(value=39, units='J')
        q2 = Quantity(value=5, units='s')
        q4 = Quantity(value=0, units='m')
        
        q3 = q1/q2
        self.assertAlmostEqual(q3.value,7.8)
        self.assertEqual(q3.units.dims,Dimensions(M=1,L=2,T=-3))
        q3 = q2/q1
        self.assertAlmostEqual(q3.value,0.128205128205)
        self.assertEqual(q3.units.dims,Dimensions(M=-1,L=-2,T=3))
        
        with self.assertRaises(ZeroDivisionError) as cm:
            q3 = q1/q4
        
    
    def test_quantity_addition(self):
        q1 = Quantity(value = 20.5, units = 'J')
        q2 = Quantity(value = 30.7, units = 'kg.m2.s-2')
        q3 = Quantity(value = 5.1, units = 'K')
        
        q4 = q1+q2
        self.assertEqual(q4.value,51.2)
        self.assertEqual(q4.units.dims,Dimensions(M=1,L=2,T=-2))
        
        with self.assertRaises(UnitsError) as cm:
            q4 = q1 + q3
        
    def test_quantity_subtraction(self):
        q1 = Quantity(value = 20.5, units = 'J')
        q2 = Quantity(value = 30.7, units = 'kg.m2.s-2')
        q3 = Quantity(value = 5.1, units = 'K')
        
        q4 = q1-q2
        self.assertEqual(q4.value,-10.2)
        self.assertEqual(q4.units.dims,Dimensions(M=1,L=2,T=-2))
        
        with self.assertRaises(UnitsError) as cm:
            q4 = q1 - q3
    
    def test_quantity_exponent(self):
        q1 = Quantity(value = 1.2, units = 'J')
        q2 = Quantity(value = -5, units = 's')
        
        q3 = q1**4
        q4 = q2**-1
        self.assertEqual(q3.value,2.0736)
        self.assertEqual(q3.units.dims,Dimensions(M=4,L=8,T=-8))
        self.assertEqual(q4.value, -0.2)
        self.assertEqual(q4.units.dims,Dimensions(T=-1))
        
        q5 = q1**0
        self.assertEqual(q5.value,1)
        self.assertNotEqual(q1.units.dims,q5.units.dims)
    
    def test_quantity_conversion(self):
        #q1 = Quantity(value = 200, units = 'J')
        #q2 = q1.convert_units_to('eV')
        #self.assertAlmostEqual(q2.value,1.2483019242e+21,places=2)
        pass
        
    def test_quantity_html(self):
        q1 = Quantity(name = 'E', value = 1.2, units = 'J')
        q2 = Quantity(value = -5, units = 's', sd = 0.3)
        q3 = Quantity(value = 30.7, units = 'kg.m2.s-2')
        q4 = Quantity(value = 22.4,units = 'm/s')
        
        self.assertEqual(q1.html_str, 'E = 1.2 J')
        self.assertEqual(q2.html_str, '-5 ± 0.3 s')
        self.assertEqual(q3.html_str, '30.7 kg m<sup>2</sup> s<sup>-2</sup>')
        self.assertEqual(q4.html_str, '22.4 m s<sup>-1</sup>')
    
if __name__ == '__main__':
    unittest.main()
