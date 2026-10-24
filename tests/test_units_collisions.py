# test_units_collisions.py
#
# Copyright (C) 2012-2016 Christian Hill
#
# Version 1.0
# Check for conflicts amongst derived and base units. For example, inches
# can't be called "in" because the milli-inch, "min" conflicts with the use
# of "min" for minutes.

import unittest

from pyqn.atom_unit import AtomUnit
from pyqn.base_unit import base_unit_stems, base_units
from pyqn.si import si_prefixes


class ConflictsCheck(unittest.TestCase):
    """
    Check that there are no conflicts amongst the allowed derived and
    base units.

    """

    def test_units_conflicts(self):
        prefixes = [""]
        prefixes.extend(si_prefixes)
        seen_units = []
        for base_unit_group in base_units:
            for base_unit in base_unit_group[1]:
                for si_prefix in prefixes:
                    this_unit = f"{si_prefix}{base_unit.stem}"
                    self.assertEqual(
                        this_unit in seen_units,
                        False,
                        f"Clash with unit: {this_unit}",
                    )
                    seen_units.append(this_unit)

    def test_prefixed_units_parse(self):
        """Check that every prefix and stem parses back to that prefix and stem."""
        for stem in base_unit_stems:
            if stem == "1":
                continue
            for si_prefix in [None, *si_prefixes]:
                s_unit = (si_prefix or "") + stem
                atom_unit = AtomUnit.parse(s_unit)
                self.assertEqual(
                    (atom_unit.prefix, atom_unit.base_unit.stem),
                    (si_prefix, stem),
                    f"Misparsed unit: {s_unit}",
                )


if __name__ == "__main__":
    unittest.main()
