#! -*- coding: utf-8 -*-
# test_units_collisions.py
# Version 0.1a
# Check for conflicts amongst derived and base units. For example, inches
# can't be called "in" because the milli-inch, "min" conflicts with the use
# of "min" for minutes.

import unittest
from units import Units
from base_unit import base_units
from si import si_prefixes

class ConflictsCheck(unittest.TestCase):
    """
    Check that there are no conflicts amongst the allowed derived and
    base units.

    """

    def test_units_conflicts(self):
        seen_units = []
        for base_unit_group in base_units:
            for base_unit in base_unit_group[1]:
                prefixes = ['']; prefixes.extend(si_prefixes)
                for si_prefix in prefixes:
                    this_unit = '%s%s' % (si_prefix, base_unit.stem)
                    self.assertEqual(this_unit in seen_units, False,
                        'Clash with unit: %s' % this_unit)
                    seen_units.append(this_unit)

if __name__ == '__main__':
    unittest.main()
