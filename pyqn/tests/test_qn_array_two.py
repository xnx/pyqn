import unittest
from ..qn_array_two import qnArrayTwo, qnArrayTwoError
from ..units import Units
from ..quantity import Quantity
import numpy as np

class qnArrayTwoTest(unittest.TestCase):
    def test_qn_array_two_init(self):
        vals1 = [1,2,3,4]
        qnarr1 = qnArrayTwo(vals1,units='m')
        self.assertEqual(qnarr1.units, Units('m'))
        for i in range(len(vals1)):
            self.assertEqual(qnarr1[i], vals1[i])
            
        vals2 = [-10,-20,-30,0,5]
        qnarr2 = qnArrayTwo(vals2, units = 'J')
        self.assertEqual(qnarr2.units, Units('J'))
        for i in range(len(vals2)):
            self.assertEqual(qnarr2[i], vals2[i])

    def test_qn_array_two_add(self):
        vals1 = [1,2,3,4]
        qnarr1 = qnArrayTwo(vals1,units='m')
        
        vals2 = [2,3,4,1]
        qnarr2 = qnArrayTwo(vals2, units='m')
        
        qnarr3 = qnarr1 + qnarr2
        self.assertEqual(qnarr3.units, qnarr1.units)
        for i in range(len(vals2)):
            self.assertEqual(qnarr3[i], vals1[i]+vals2[i])
            
        vals3 = [1,1,1]
        qnarr3 = qnArrayTwo(vals3, units='m')
        with self.assertRaises(ValueError) as e:
            qnarr = qnarr1 + qnarr3

        q1 = Quantity(value = 10, units = 'm')
        qnarr4 = qnarr1 + q1
        self.assertEqual(qnarr4.units, Units('m'))
        for i in range(len(qnarr4)):
            self.assertEqual(qnarr4[i], qnarr1[i]+10)
            
    def test_qn_array_two_sub(self):
        vals1 = [1,2,3,4]
        qnarr1 = qnArrayTwo(vals1,units='m')
        
        vals2 = [2,3,4,1]
        qnarr2 = qnArrayTwo(vals2, units='m')
        
        qnarr3 = qnarr1 - qnarr2
        for i in range(len(vals2)):
            self.assertEqual(qnarr3[i], vals1[i]-vals2[i])
            
        vals3 = [1,1,1]
        qnarr3 = qnArrayTwo(vals3, units='m')
        with self.assertRaises(ValueError) as e:
            qnarr = qnarr1 - qnarr3

        q1 = Quantity(value = 10, units = 'm')
        qnarr4 = qnarr1 - q1
        self.assertEqual(qnarr4.units, Units('m'))
        for i in range(len(qnarr4)):
            self.assertEqual(qnarr4[i], qnarr1[i]-10)

    def test_qn_array_two_mul(self):
        vals1 = [1,2,3,4]
        qnarr1 = qnArrayTwo(vals1,units='m')
        
        vals2 = [2,3,4,1]
        qnarr2 = qnArrayTwo(vals2, units='s')
        
        qnarr3 = qnarr1 * qnarr2
        self.assertEqual(qnarr3.units, Units('m.s'))
        for i in range(len(vals2)):
            self.assertEqual(qnarr3[i], vals1[i]*vals2[i])
            
        vals3 = [1,1,1]
        qnarr3 = qnArrayTwo(vals3, units='m')
        with self.assertRaises(ValueError) as e:
            qnarr = qnarr1 * qnarr3

        q1 = Quantity(value = 10, units = 'J')
        qnarr4 = qnarr1 * q1
        self.assertEqual(qnarr4.units, Units('m.J'))
        for i in range(len(qnarr4)):
            self.assertEqual(qnarr4[i], qnarr1[i]*10)

    def test_qn_array_two_mul(self):
        vals1 = [1,2,3,4]
        qnarr1 = qnArrayTwo(vals1,units='m')
        
        vals2 = [2,3,4,1]
        qnarr2 = qnArrayTwo(vals2, units='s')
        
        qnarr3 = qnarr1 / qnarr2
        self.assertEqual(qnarr3.units, Units('m.s-1'))
        for i in range(len(vals2)):
            self.assertEqual(qnarr3[i], vals1[i]/vals2[i])
            
        vals3 = [1,1,1]
        qnarr3 = qnArrayTwo(vals3, units='m')
        with self.assertRaises(ValueError) as e:
            qnarr = qnarr1 / qnarr3

        q1 = Quantity(value = 10, units = 'J')
        qnarr4 = qnarr1 / q1
        self.assertEqual(qnarr4.units, Units('m.J-1'))
        for i in range(len(qnarr4)):
            self.assertEqual(qnarr4[i], qnarr1[i]/10)

    def test_qn_array_two_pow(self):
        vals1 = [1,2,3,4]
        qnarr1 = qnArrayTwo(vals1, units = 'm')

        qnarr2 = qnarr1 ** 2
        self.assertEqual(qnarr2.units, Units('m2'))
        for i in range(len(vals1)):
            self.assertEqual(qnarr2[i], vals1[i]**2)

    #def test_qn_array_eq(self):
    #    qnarr1 = qnArrayTwo([1,1,1],units = 'm', sd = [0.1,0.1,0.1])
    #    qnarr2 = qnArrayTwo([1,1,1],units = 'm')
    #    qnarr3 = qnArrayTwo([1,2,3],units = 'm', sd = [0.1,0.1,0.1])
    #    qnarr4 = qnArrayTwo([1,2,3],units = 'J', sd = [0.1,0.1,0.1])
    #    qnarr5 = qnArrayTwo([1.0,1.0,1.0],units = Units('m'), sd = [0,0,0])

    #    self.assertFalse(qnarr1 == qnarr2)
    #    self.assertFalse(qnarr1 == qnarr3)
    #    self.assertFalse(qnarr1 == qnarr4)
    #    self.assertFalse(qnarr1 == qnarr5)

    #    self.assertFalse(qnarr2 == qnarr3)
    #    self.assertFalse(qnarr2 == qnarr4)
    #    self.assertTrue(qnarr2 == qnarr5)

    #    self.assertFalse(qnarr3 == qnarr4)
    #    self.assertFalse(qnarr3 == qnarr5)

    #    self.assertFalse(qnarr4 == qnarr5)

    def test_qn_array_two_html(self):
        vals = [1,2,3,4]
        qnarr = qnArrayTwo(vals, units = 'm')

        #self.assertEqual(qnarr.html_str, '1 m, 2 m, 3 m, 4 m')

if __name__ == '__main__':
    unittest.main()
