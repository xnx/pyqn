# test_units_collisions.py
#
# Copyright (C) 2012-2016 Christian Hill
#
# Version 1.0
# Check for conflicts amongst derived and base units. For example, inches
# can't be called "in" if the inch takes prefixes, because the milli-inch,
# "min", would conflict with the use of "min" for minutes.

import unittest
import warnings

from pyqn.atom_unit import AtomUnit
from pyqn.base_unit import base_unit_stems, base_units, metric, v2_stems
from pyqn.si import si_prefixes
from pyqn.units import UnitsError


def spellings(base_unit):
    """
    Return the (prefix, stem) pairs that parse to base_unit: those its rules
    allow and, until pyqn 3.0, the deprecated ones that pyqn 2.0 allowed.

    """

    if base_unit.stem == "1":
        # the unity 'unit' takes no prefix
        return [(None, "1")]
    prefixes = set(base_unit.prefixes)
    if base_unit.stem in v2_stems:
        prefixes |= metric
    return [(None, base_unit.stem)] + [(p, base_unit.stem) for p in prefixes]


class ConflictsCheck(unittest.TestCase):
    """
    Check that there are no conflicts amongst the allowed derived and
    base units.

    """

    def test_units_conflicts(self):
        seen_units = set()
        for base_unit_group in base_units:
            for base_unit in base_unit_group[1]:
                for si_prefix, stem in spellings(base_unit):
                    this_unit = f"{si_prefix or ''}{stem}"
                    self.assertNotIn(
                        this_unit, seen_units, f"Clash with unit: {this_unit}"
                    )
                    seen_units.add(this_unit)

    def test_prefixed_units_parse(self):
        """
        Check that every prefix and stem parses back to that prefix and stem:
        silently if the unit takes the prefix, with a DeprecationWarning if it
        took it in pyqn 2.0, and otherwise not at all.

        """

        for stem, base_unit in base_unit_stems.items():
            if stem == "1":
                continue
            for si_prefix in [None, *si_prefixes]:
                s_unit = (si_prefix or "") + stem
                if si_prefix is not None and s_unit in base_unit_stems:
                    # e.g. 'hbar' is a unit, not the hectobar
                    self.assertNotIn(si_prefix, base_unit.prefixes, s_unit)
                    continue
                if si_prefix is None or si_prefix in base_unit.prefixes:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        atom_unit = AtomUnit.parse(s_unit)
                elif stem in v2_stems and si_prefix in metric:
                    with self.assertWarns(DeprecationWarning, msg=s_unit):
                        atom_unit = AtomUnit.parse(s_unit)
                else:
                    with self.assertRaises(UnitsError, msg=s_unit):
                        AtomUnit.parse(s_unit)
                    continue
                self.assertEqual(
                    (atom_unit.prefix, atom_unit.base_unit.stem),
                    (si_prefix, stem),
                    f"Misparsed unit: {s_unit}",
                )


if __name__ == "__main__":
    unittest.main()
