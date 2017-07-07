import unittest
from ..si import SIPrefix, si_prefixes

class SICheck(unittest.TestCase):
    def test_siprefix_init(self):
        pass
        
    def test_siprefixes(self):
        self.assertEqual(si_prefixes['y'].prefix, 'y')
        self.assertEqual(si_prefixes['y'].name, 'yocto')
        self.assertEqual(si_prefixes['y'].power, -24)
        self.assertEqual(si_prefixes['y'].fac, 1e-24)
        
        self.assertEqual(si_prefixes['μ'].prefix, 'μ')
        self.assertEqual(si_prefixes['μ'].name, 'micro')
        self.assertEqual(si_prefixes['μ'].power, -6)
        self.assertEqual(si_prefixes['μ'].fac, 1e-6)
        
        self.assertEqual(si_prefixes['M'].prefix, 'M')
        self.assertEqual(si_prefixes['M'].name, 'mega')
        self.assertEqual(si_prefixes['M'].power, 6)
        self.assertEqual(si_prefixes['M'].fac, 1e6)
        
    def test_si_unit_systems(self):
        pass
        
if __name__ == '__main__':
    unittest.main()
