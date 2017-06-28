import unittest
from ..qn_array import qnArray, qnArrayError
from ..quantity import Quantity
import numpy as np

class qnArrayTest(unittest.TestCase):
    def test_qnarray_init(self):
        a1 = [1,2,3,4,5]
        qnarr1 = qnArray(values = a1, units = 'm')
        for i in range(len(a1)):
            self.assertEqual(qnarr1.values[i], a1[i])
            self.assertEqual(qnarr1.nparr[i], Quantity(value=a1[i], units='m'))
        self.assertEqual(qnarr1.units, 'm')
        
        a2 = [-5,-10]
        qnarr2 = qnArray(values = np.array(a2), units = 'J')
        for i in range(len(a2)):
            self.assertEqual(qnarr2.values[i], a2[i])
            self.assertEqual(qnarr2.nparr[i], Quantity(value=a2[i], units='J'))
        self.assertEqual(qnarr2.units, 'J')
        
        with self.assertRaises(qnArrayError) as e1:
            qnarr = qnArray(values = "string")
        with self.assertRaises(qnArrayError) as e2:
            qnarr = qnArray(values = ["a", "b", "c"])
            
if __name__ == '__main__':
    unittest.main()
