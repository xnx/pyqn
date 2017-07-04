import unittest
from ..qn_array_two import qnArrayTwo
from ..units import Units

class qnArrayTwoTest(unittest.TestCase):
    def qn_array_two_init(self):
        vals1 = [1,2,3,4]
        qnarr = qnArrayTwo(vals1,units='m')
        self.assertEqual(qnarr.units, Units('m'))
        for i in range(len(vals1)):
            self.assertEqual(qnarr.values[i], vals1[i])

if __name__ == '__main__':
    unittest.main()
