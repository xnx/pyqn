import unittest
from ..qn_array_two import qnArrayTwo, qnArrayTwoError
from ..units import Units
from ..quantity import Quantity

class qnArrayTwoTest(unittest.TestCase):
    def test_qn_array_two_init(self):
        vals1 = [1,2,3,4]
        qnarr1 = qnArrayTwo(vals1,units='m')
        self.assertEqual(qnarr1.units, Units('m'))
        for i in range(len(vals1)):
            self.assertEqual(qnarr1[i], vals1[i])
            
        vals2 = [-10,-20,-30,0,5]
        sd2 = [0.1,0.2,0.3,0.4,0.1]
        qnarr2 = qnArrayTwo(vals2, units = 'J', sd = sd2)
        self.assertEqual(qnarr2.units, Units('J'))
        for i in range(len(vals2)):
            self.assertEqual(qnarr2[i], vals2[i])
            self.assertEqual(qnarr2.sd[i], sd2[i])

        with self.assertRaises(qnArrayTwoError) as e:
            qnarr = qnArrayTwo(vals2, units='m', sd = vals1)

    def test_qn_array_two_add(self):
        vals1 = [1,2,3,4]
        qnarr1 = qnArrayTwo(vals1,units='m')
        
        vals2 = [2,3,4,1]
        qnarr2 = qnArrayTwo(vals2, units='m')
        
        qnarr3 = qnarr1 + qnarr2
        for i in range(len(vals2)):
            self.assertEqual(qnarr3[i], vals1[i]+vals2[i])
            
        vals3 = [1,1,1]
        qnarr3 = qnArrayTwo(vals3, units='m')
        with self.assertRaises(qnArrayTwoError) as e:
            qnarr = qnarr1 + qnarr3

        q1 = Quantity(value = 10, units = 'm')
        qnarr4 = qnarr1+q1
        self.assertEqual(qnarr4.units, Units('m'))
        for i in range(len(qnarr4)):
            self.assertEqual(qnarr4[i], qnarr1[i]+10)

        with self.assertRaises(qnArrayTwoError) as e:
            qnarr = qnarr1 + 2
            
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
        with self.assertRaises(qnArrayTwoError) as e:
            qnarr = qnarr1 - qnarr3

        q1 = Quantity(value = 10, units = 'm')
        qnarr4 = qnarr1 - q1
        self.assertEqual(qnarr4.units, Units('m'))
        for i in range(len(qnarr4)):
            self.assertEqual(qnarr4[i], qnarr1[i]-10)

        with self.assertRaises(qnArrayTwoError) as e:
            qnarr = qnarr1 - 2

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
        with self.assertRaises(qnArrayTwoError) as e:
            qnarr = qnarr1 * qnarr3

        q1 = Quantity(value = 10, units = 'J')
        qnarr4 = qnarr1 * q1
        self.assertEqual(qnarr4.units, Units('m.J'))
        for i in range(len(qnarr4)):
            self.assertEqual(qnarr4[i], qnarr1[i]*10)

        with self.assertRaises(qnArrayTwoError) as e:
            qnarr = qnarr1 * 2

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
        with self.assertRaises(qnArrayTwoError) as e:
            qnarr = qnarr1 / qnarr3

        q1 = Quantity(value = 10, units = 'J')
        qnarr4 = qnarr1 / q1
        self.assertEqual(qnarr4.units, Units('m.J-1'))
        for i in range(len(qnarr4)):
            self.assertEqual(qnarr4[i], qnarr1[i]/10)

        with self.assertRaises(qnArrayTwoError) as e:
            qnarr = qnarr1 / 2

if __name__ == '__main__':
    unittest.main()
