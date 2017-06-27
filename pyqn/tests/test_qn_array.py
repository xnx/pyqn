import unittest
from ..qn_array import qnArray
from ..quantity import Quantity

class qnArrayTest(unittest.TestCase):
	def test_qnarray_init(self):
		qnarr1 = qnArray(values = [1,2,3,4,5], units = 'm')
		qnarr2 = qnArray(values = np.array([-5,-10], units = 'J')
		
		with self.assertRaises(SOMETHINGERROR) as e:
			qnarr = qnArray(values = "string")
			
