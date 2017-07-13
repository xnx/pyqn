import unittest
from ..atom_unit import AtomUnit
from ..base_unit import BaseUnit, base_units
from ..dimensions import d_length, d_energy
from ..si import SIPrefix

class AtomUnitCheck(unittest.TestCase):
    def test_atom_unit_init(self):
        au1 = AtomUnit('m', BaseUnit('m', 'metre', 'length', 1., '', 'm', d_length), -2)
        self.assertEqual(au1.base_unit, BaseUnit('m', 'metre', 'length', 1., '', 'm', d_length))
        self.assertEqual(au1.exponent, -2)
        self.assertEqual(au1.si_prefix, SIPrefix('m', 'milli', -3))
        self.assertEqual(au1.si_fac, 1000000.0)
        
    def test_atom_unit_parse(self):
        au1 = AtomUnit.parse('μJ')
        self.assertEqual(au1.base_unit, BaseUnit('J','','',1,'','',d_energy))
        self.assertEqual(au1.exponent, 1)
        self.assertEqual(au1.si_prefix, SIPrefix('μ', 'micro', -6))
        self.assertEqual(au1.si_fac, 1e-6)

    def test_atom_unit_pow(self):
        au1 = AtomUnit.parse('mm')
        au2 = au1 ** 3
        self.assertEqual(au2.base_unit, BaseUnit('m', 'metre', 'length', 1., '', 'm', d_length), -2)
        self.assertEqual(au2.exponent, 3)
        self.assertEqual(au2.si_prefix, SIPrefix('m', 'milli', -3))
        self.assertEqual(au2.si_fac, 1e-9)

if __name__ == '__main__':
    unittest.main()
