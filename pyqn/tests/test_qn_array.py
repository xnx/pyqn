import unittest
from ..qn_array import qnArray
from ..quantity import Quantity
import numpy as np

class qnArrayTest(unittest.TestCase):
    def test_qnarray_init(self):
        qnarr1 = qnArray(values = [1,2,3,4,5], units = 'm')
        self.assertEqual(qnarr1.values = [1,2,3,4,5])
        self.assertEqual(qnarr1.units = 'm')
        self.assertEqual(qnarr1.nparr = np.array([Quantity(value=1,units='m'),
                                                  Quantity(value=2,units='m'),
                                                  Quantity(value=3,units='m'),
                                                  Quantity(value=4,units='m'),
                                                  Quantity(value=5,units='m')])
        
        qnarr2 = qnArray(values = np.array([-5,-10], units = 'J')
        self.assertEqual(qnarr2.values = [-5,-10])
        self.assertEqual(qnarr2.units = 'J')
        self.assertEqual(qnarr2.nparr = np.array([Quantity(value=-5,units='J'),
												  Quantity(value=-10,units='J')])
        
        with self.assertRaises(SOMETHINGERROR) as e1:
            qnarr = qnArray(values = "string")
        with self.assertRaises(SOMETHINGERROR) as e2:
            qnarr = qnArray(values = ["a", "b", "c"])
            
