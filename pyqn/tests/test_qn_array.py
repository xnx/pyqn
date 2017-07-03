import unittest
from ..qn_array import qnArray, qnArrayError
from ..quantity import Quantity
import numpy as np

class qnArrayTest(unittest.TestCase):
    def test_qnarray_init(self):
        a1 = [-1,-2,-3,-4,-5]
        qnarr1 = qnArray(values = a1, units = 'm')
        for i in range(len(a1)):
            self.assertEqual(qnarr1.values[i], a1[i])
            self.assertEqual(qnarr1.nparr[i], a1[i])
        self.assertEqual(qnarr1.units_str, 'm')
        
        a2 = [5,10]
        qnarr2 = qnArray(values = np.array(a2), units = 'J')
        for i in range(len(a2)):
            self.assertEqual(qnarr2.values[i], a2[i])
            self.assertEqual(qnarr2.nparr[i], a2[i])
        self.assertEqual(qnarr2.units_str, 'J')
        
        with self.assertRaises(qnArrayError) as e1:
            qnarr = qnArray(values = "string")
        with self.assertRaises(qnArrayError) as e2:
            qnarr = qnArray(values = ["a", "b", "c"])
            
    def test_qnarray_mult(self):
        a1 = [-1,-2,-3,-4,-5]
        qnarr1 = qnArray(values = a1, units = 'm')
        q1 = Quantity(value = 2, units = 's')
        result1 = qnarr1*q1
        self.assertEqual(result1.units_str, 'm.s')
        for i in range(len(a1)):
            self.assertEqual(result1.nparr[i], 2*a1[i])
            
        a2 = [5,10]
        qnarr2 = qnArray(values = np.array(a2), units = 'J')
        q2 = Quantity(value=0.2, units = 'm-1')
        result2 = qnarr2*q2
        self.assertEqual(result2.units_str, 'J.m-1')
        for i in range(len(a2)):
            self.assertEqual(result2.nparr[i], 0.2*a2[i])
            
    def test_qnarray_div(self):
        a1 = [-1,-2,-3,-4,-5]
        qnarr1 = qnArray(values = a1, units = 'm')
        q1 = Quantity(value = 2, units = 's')
        result1 = qnarr1/q1
        self.assertEqual(result1.units_str, 'm.s-1')
        for i in range(len(a1)):
            self.assertEqual(result1.nparr[i], a1[i]/2)
            
if __name__ == '__main__':
    unittest.main()
