import unittest

from pyqn.units import Units


class UserDefinedUnitsCheck(unittest.TestCase):
    """Unit tests for user-defined units."""

    @unittest.skip("user-defined dimensionless units are not implemented yet")
    def test_dimensionless_units(self):
        Units("cm-2.molec-1", dimensionless=("molec",))


if __name__ == "__main__":
    unittest.main()
