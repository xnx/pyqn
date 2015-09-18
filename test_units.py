# -*- coding: utf-8 -*-
# test_units.py
# Version 0.1a
# Unit tests for the Units class.

import unittest
from units import Units

class UnitsCheck(unittest.TestCase):
    """Unit tests for the Units class."""

    def test_units_algebra(self):
        u1 = Units('m.s-1')
        self.assertTrue(u1.has_units())
        u2 = Units('cm.hr-1/m')

        self.assertEqual(str(u1), 'm.s-1')
        self.assertEqual(str(u2), 'cm.hr-1.m-1')

        self.assertEqual(u1*u2, Units('cm.s-1.hr-1'))
        self.assertEqual(u1/u2, Units('m2.s-1.cm-1.hr'))
        self.assertEqual(u2/u1, Units('cm.hr-1.s/m2'))

        u3 = Units('J.A-1')
        u4 = Units('J.A-1')
        u3_over_u4 = u3 / u4
        self.assertFalse(u3_over_u4.has_units())

    def test_unicode_units(self):
        u1 = Units(u'kΩ')
        self.assertEqual(unicode(u1), u'kΩ')

if __name__ == '__main__':
    unittest.main()
