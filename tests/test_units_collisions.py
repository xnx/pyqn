# test_units_collisions.py
#
# Copyright (C) 2012-2016 Christian Hill
#
# Version 1.0
# Check for conflicts amongst derived and base units. For example, inches
# can't be called "in" if the inch takes prefixes, because the milli-inch,
# "min", would conflict with the use of "min" for minutes. The Domain
# constructor checks for such conflicts in every domain.

import unittest
import warnings

from pyqn.atom_unit import AtomUnit
from pyqn.base_unit import base_unit_stems, metric, v2_stems
from pyqn.domains import Domain, everything
from pyqn.si import si_prefixes
from pyqn.units import UnitsError


class ConflictsCheck(unittest.TestCase):
    """
    Check that there are no conflicts amongst the allowed derived and
    base units.

    """

    def test_units_conflicts(self):
        # making the default domain again runs the Domain constructor's check
        Domain(
            "everything",
            units=everything.units,
            spellings=dict(everything.spellings),
            v2_prefixes=True,
        )

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
