import unittest
from ..si import SIPrefix, si_prefixes

class SICheck(unittest.TestCase):
    def test_siprefix_init(self):
        s = SIPrefix('PREFIX','NAME',2.5)
        self.assertEqual(s.prefix, 'PREFIX')
        self.assertEqual(s.name, 'NAME')
        self.assertEqual(s.power, 2.5)
        self.assertEqual(s.fac, 10**2.5)
        
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
        
    def test_siprefix_eq(self):
        self.assertTrue(SIPrefix('PREFIX','NAME',2.0) == SIPrefix('PREFIX','',2))
        self.assertFalse(SIPrefix('P','NAME',3) == SIPrefix('P','NAME',-3))
        
if __name__ == '__main__':
    unittest.main()
