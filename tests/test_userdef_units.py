import unittest
from fractions import Fraction
from pyqn.units import Units
from pyqn.dimensions import Dimensions, d_energy


class UserDefinedUnitsCheck(unittest.TestCase):
    """Unit tests for user-defined units."""

    def test_dimensionless_units(self):
        return
        u1 = Units("cm-2.molec-1", dimensionless=("molec",))


if __name__ == "__main__":
    unittest.main()
