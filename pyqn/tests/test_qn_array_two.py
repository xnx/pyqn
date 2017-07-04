import unittest
from ..qn_array_two import qnArrayTwo
from ..units import Units

class qnArrayTwoTest(unittest.TestCase):
    def qn_array_two_init(self):
        vals1 = [1,2,3,4]
        qnarr1 = qnArrayTwo(vals1,units='m')
        self.assertEqual(qnarr1.units, Units('m'))
        for i in range(len(vals1)):
            self.assertEqual(qnarr1.values[i], vals1[i])
            
        vals2 = [-10,-20,-30,0,5]
        sd2 = [0.1,0.2,0.3,0.4,0.1]
        qnarr2 = qnArrayTwo(vals2, units = 'J', sd = sd2)
        self.assertEqual(qnarr2.units, Units('J'))
        for i in range(len(vals2)):
            self.assertEqual(qnarr2.values[i], vals2[i])
            self.assertEqual(qnarr2.sd[i], sd2[i])

if __name__ == '__main__':
    unittest.main()
