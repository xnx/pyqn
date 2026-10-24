"""Unit tests of basic functionality for the Units class."""

import math
import unittest
from fractions import Fraction
import pytest
from pyqn.units import Units
from pyqn.dimensions import Dimensions, d_energy

e = 1.602176634e-19


def approx(expected, rel=1.0e-6):
    """
    Compare with a relative tolerance only: pytest.approx's default absolute
    tolerance (1e-12) would accept any two very small factors.

    """
    return pytest.approx(expected, rel=rel, abs=0)


class UnitsCheck(unittest.TestCase):
    """Unit tests for the Units class."""

    def test_units_algebra(self):
        u1 = Units("m.s-1")
        self.assertTrue(u1.has_units())
        u2 = Units("cm.hr-1/m")

        self.assertEqual(str(u1), "m.s-1")
        self.assertEqual(str(u2), "cm.hr-1.m-1")

        self.assertEqual(u1 * u2, Units("cm.s-1.hr-1"))
        self.assertEqual(u1 / u2, Units("m2.s-1.cm-1.hr"))
        self.assertEqual(u2 / u1, Units("cm.hr-1.s/m2"))

        u3 = Units("J.A-1")
        u4 = Units("J.A-1")
        u3_over_u4 = u3 / u4
        self.assertFalse(u3_over_u4.has_units())

    def test_units_multiplication(self):
        u1 = Units("m.s-1")
        u2 = u1 * 1

        self.assertEqual(u2, u1)
        self.assertNotEqual(id(u1), id(u2))

        u2 = 1 * u1
        self.assertEqual(u2, u1)
        self.assertNotEqual(id(u1), id(u2))

        u1 = Units("m.s-1")
        u2 = Units("J")
        self.assertEqual("kg.m.s-1" * u1, u2)

        u1 = Units("m.s-1")
        u2 = Units("J")
        self.assertEqual(u1 * "kg.m.s-1", u2)

        u1 = Units("mm2")
        u2 = Units("g/m2")
        u3 = u1 * u2
        self.assertEqual(u3.to_si(), approx(1.0e-9))
        self.assertTupleEqual(u3.get_dims().dims, (0, 1, 0, 0, 0, 0, 0))

        u4 = Units("J/s")
        u5 = u4 * u2
        self.assertEqual(u5.to_si(), approx(1.0e-3))
        self.assertEqual(u5.get_dims().dims, (0, 2, -3, 0, 0, 0, 0))

    def test_units_division(self):
        u1 = Units("eV.mm")
        u2 = Units("K.cm")

        u3 = u1 / u2
        self.assertEqual(u3.get_dims().dims, (2, 1, -2, -1, 0, 0, 0))
        self.assertEqual(u3.to_si(), approx(e * 1.0e-3 / 1.0e-2))

        u4 = Units("s-1")
        u5 = u2 / ("eV" * u4)
        self.assertEqual(u5.get_dims(), Dimensions(T=1, Theta=1, L=1) / d_energy)

    def test_units_power(self):
        u1 = Units("J.m")
        u2 = u1**2
        self.assertEqual(u2.get_dims(), Dimensions(M=2, L=6, T=-4))

        u3 = u1 ** (-1)
        self.assertEqual(u3.get_dims(), Dimensions(M=-1, L=-3, T=2))

        u4 = u1 ** (-2)
        self.assertEqual(u4.get_dims(), Dimensions(M=-2, L=-6, T=4))

        u5 = u1**0
        self.assertEqual(u5.get_dims(), Dimensions())

    def test_units_algebra_dimensions(self):
        u1 = Units("m")
        u2 = Units("m.s-1")
        u3 = u1 * u2
        u4 = Units("m2.s-1")
        self.assertEqual(u3.dims, u4.dims)

        u3 = u1 / u2
        u4 = Units("s")
        self.assertEqual(u3.dims, u4.dims)

    def test_unicode_units(self):
        u1 = Units("kΩ")
        self.assertEqual(str(u1), "kΩ")

    def test_html(self):
        u1 = Units("m.s-1")
        self.assertEqual(u1.html, "m s<sup>-1</sup>")
        u2 = Units("μs.J/m3")
        self.assertEqual(u2.html, "μs J m<sup>-3</sup>")

    def test_units_with_rational_powers(self):
        u1 = Units("m-3.Pa-1/2")
        u2 = Units("cm-3.bar-1/2")
        self.assertEqual(str(u1), "m-3.Pa-1/2")
        self.assertEqual(u1.dims, u2.dims)
        self.assertEqual(u1.html, "m<sup>-3</sup> Pa<sup>-1/2</sup>")

        self.assertEqual(u1.dims, Dimensions(M=-0.5, T=1, L=-2.5))

        u3 = Units("kg1/2.m-3/2")
        self.assertEqual(str(u3), "kg1/2.m-3/2")
        self.assertEqual(u3.html, "kg<sup>1/2</sup> m<sup>-3/2</sup>")

    def test_rational_units_algebra(self):
        u1 = Units("m-3.Pa-1/2")
        u2 = Units("m3")
        u3 = u1 * u2
        self.assertEqual(str(u3), "Pa-1/2")

        u4 = Units("Pa-2/3.s4")
        u5 = u1 / u4
        self.assertEqual(u5, Units("m-3.Pa1/6.s-4"))
        u6 = Units("m-3.s4.Pa1/6")
        self.assertEqual(str(u6), "m-3.s4.Pa1/6")
        self.assertEqual(
            u6.dims,
            Dimensions(M=Fraction("1/6"), T=Fraction("11/3"), L=Fraction("-19/6")),
        )

    def test_units_equality(self):
        self.assertEqual(Units("J"), Units("kg.m2.s-2"))
        self.assertEqual(Units("cm.cm"), Units("cm2"))
        self.assertEqual(Units("fm"), Units("fm"))
        self.assertEqual(Units("1"), 1)
        self.assertEqual(Units("m/m"), 1)
        self.assertNotEqual(Units("m"), Units("cm"))
        self.assertNotEqual(Units("m"), Units("s"))

    def test_product_leaves_operands_unchanged(self):
        u1 = Units("m")
        u2 = Units("cm.cm")
        u1 * u2
        u1 / u2
        self.assertEqual(str(u1), "m")
        self.assertEqual(str(u2), "cm.cm")

    def test_merged_exponents_factor(self):
        self.assertEqual((Units("cm") * Units("cm")).to_si(), approx(1.0e-4))
        self.assertEqual((Units("km2") / Units("km")).to_si(), approx(1.0e3))
        self.assertEqual(Units("km.hr-1/hr").to_si(), approx(1.0e3 / 3600**2))

    def test_equality_of_small_factors(self):
        self.assertNotEqual(Units("pm"), Units("fm"))
        self.assertNotEqual(Units("eV"), Units("E_h"))
        self.assertNotEqual(Units("u"), Units("m_e"))

    def test_dimensionless_units(self):
        self.assertNotEqual(Units("deg"), Units("rad"))
        self.assertNotEqual(Units("deg"), 1)
        self.assertEqual((Units("m") * Units("deg")).to_si(), approx(math.pi / 180))

    def test_reflected_division(self):
        self.assertEqual(1 / Units("s"), Units("s-1"))
        self.assertEqual("m" / Units("s"), Units("m.s-1"))

    def test_zero_power(self):
        u1 = Units("J.m") ** 0
        self.assertFalse(u1.has_units())
        self.assertEqual(str(u1), "")

    def test_fractional_dimensions_str(self):
        self.assertEqual(str(Units("Pa-1/2").dims), "L1/2.M-1/2.T")


if __name__ == "__main__":
    unittest.main()
