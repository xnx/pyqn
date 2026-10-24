import math
import unittest
import pytest
from pyqn.quantity import Quantity, QuantityError
from pyqn.dimensions import Dimensions, d_energy
from pyqn.units import Units, UnitsError

e = 1.602176634e-19


class QuantityManipulations(unittest.TestCase):
    def test_quantity_init(self):
        pass

    def test_quantity_multiplication(self):
        q1 = Quantity(value=22.4, units="m/s")
        q2 = Quantity(value=2, units="s")

        q3 = q1 * q2
        self.assertAlmostEqual(q3.value, 44.8)
        self.assertEqual(q3.units.dims.dims, (1, 0, 0, 0, 0, 0, 0))

    def test_quantity_division(self):
        q1 = Quantity(value=39, units="J")
        q2 = Quantity(value=5, units="s")
        q4 = Quantity(value=0, units="m")

        q3 = q1 / q2
        self.assertAlmostEqual(q3.value, 7.8)
        self.assertEqual(q3.units.dims, Dimensions(M=1, L=2, T=-3))
        q3 = q2 / q1
        self.assertAlmostEqual(q3.value, 0.128205128205)
        self.assertEqual(q3.units.dims, Dimensions(M=-1, L=-2, T=3))

        with self.assertRaises(ZeroDivisionError) as cm:
            q3 = q1 / q4

    def test_quantity_addition(self):
        q1 = Quantity(value=20.5, units="J")
        q2 = Quantity(value=30.7, units="kg.m2.s-2")
        q3 = Quantity(value=5.1, units="K")

        q4 = q1 + q2
        self.assertEqual(q4.value, 51.2)
        self.assertEqual(q4.units.dims, Dimensions(M=1, L=2, T=-2))

        with self.assertRaises(UnitsError) as cm:
            q4 = q1 + q3

    def test_quantity_subtraction(self):
        q1 = Quantity(value=20.5, units="J")
        q2 = Quantity(value=30.7, units="kg.m2.s-2")
        q3 = Quantity(value=5.1, units="K")

        q4 = q1 - q2
        self.assertEqual(q4.value, -10.2)
        self.assertEqual(q4.units.dims, Dimensions(M=1, L=2, T=-2))

        with self.assertRaises(UnitsError) as cm:
            q4 = q1 - q3

    def test_quantity_exponent(self):
        q1 = Quantity(value=1.2, units="J")
        q2 = Quantity(value=-5, units="s")

        q3 = q1**4
        q4 = q2**-1
        self.assertEqual(q3.value, 2.0736)
        self.assertEqual(q3.units.dims, Dimensions(M=4, L=8, T=-8))
        self.assertEqual(q4.value, -0.2)
        self.assertEqual(q4.units.dims, Dimensions(T=-1))

        q5 = q1**0
        self.assertEqual(q5.value, 1)
        self.assertNotEqual(q1.units.dims, q5.units.dims)

    def test_quantity_conversion(self):
        q1 = Quantity(value=200, units="J", sd=2)
        q2 = q1.convert_units_to("eV")
        self.assertEqual(q2.value, pytest.approx(200 / e))
        self.assertEqual(q2.sd, pytest.approx(2 / e))
        self.assertEqual(q2.units, Units("eV"))

    def test_scalar_multiplication(self):
        q1 = Quantity(value=10, units="m", sd=1)
        q2 = q1 * 2
        self.assertAlmostEqual(q2.value, 20)
        self.assertAlmostEqual(q2.sd, 2)
        self.assertEqual(q2.units, Units("m"))

    def test_reflected_multiplication(self):
        q1 = Quantity(value=10, units="m", sd=1)
        q2 = 2 * q1
        self.assertAlmostEqual(q2.value, 20)
        self.assertAlmostEqual(q2.sd, 2)

    def test_scalar_division(self):
        q1 = Quantity(value=10, units="m", sd=1)
        q2 = q1 / 2
        self.assertAlmostEqual(q2.value, 5)
        self.assertAlmostEqual(q2.sd, 0.5)
        self.assertEqual(q2.units, Units("m"))

    def test_reflected_division(self):
        q1 = Quantity(value=10, units="m", sd=1)
        q2 = 2 / q1
        self.assertAlmostEqual(q2.value, 0.2)
        self.assertAlmostEqual(q2.sd, 0.02)
        self.assertEqual(q2.units, Units("m-1"))

    def test_negative_product_uncertainty(self):
        q1 = Quantity(value=10, units="m", sd=1)
        q2 = Quantity(value=-2, units="s", sd=0.1)
        self.assertAlmostEqual((q1 * q2).sd, 20 * math.hypot(0.1, 0.05))
        self.assertAlmostEqual((q1 / q2).sd, 5 * math.hypot(0.1, 0.05))

    def test_product_with_exact_operand(self):
        q1 = Quantity(value=10, units="m", sd=1)
        q2 = Quantity(value=2, units="s", sd=0)
        self.assertIsNotNone((q1 * q2).sd)
        self.assertAlmostEqual((q1 * q2).sd, 2)
        self.assertAlmostEqual((q1 / q2).sd, 0.5)

    def test_product_with_zero_value(self):
        q1 = Quantity(value=0, units="m", sd=1)
        q2 = Quantity(value=2, units="s", sd=0.1)
        self.assertAlmostEqual((q1 * q2).sd, 2)
        self.assertAlmostEqual((q1 / q2).sd, 0.5)

    def test_add_dimensionless_quantities(self):
        q1 = Quantity(value=90, units="deg")
        q2 = Quantity(value=1, units="rad")
        with self.assertRaises(UnitsError):
            q1 + q2

    def test_add_quantities_with_small_units(self):
        q1 = Quantity(value=1, units="pm")
        q2 = Quantity(value=1, units="fm")
        with self.assertRaises(UnitsError):
            q1 + q2


class QuantityStringsCheck(unittest.TestCase):
    """Unit tests for parsing and formatting Quantity objects as strings."""

    def test_value_as_str(self):
        self.assertEqual(Quantity(value=1.2345, sd=0.0012).value_as_str(), "1.2345(12)")
        self.assertEqual(Quantity(value=-0.412, sd=0.04).value_as_str(), "-0.412(40)")
        self.assertEqual(
            Quantity(value=17324000, sd=380).value_as_str(), "1.732400(38)e7"
        )

    def test_value_as_str_large_sd(self):
        self.assertEqual(Quantity(value=1234.4, sd=12).value_as_str(), "1234(12)")
        self.assertEqual(Quantity(value=1.234, sd=0.01).value_as_str(), "1.234(10)")
        self.assertEqual(Quantity(value=1.234, sd=0.0996).value_as_str(), "1.23(10)")

    def test_as_str(self):
        q1 = Quantity(name="x", value=1.2345, sd=0.0012, units="m")
        self.assertEqual(q1.as_str(), "x = 1.2345(12) m")

    def test_parse(self):
        q1 = Quantity.parse("x = 1.5(2)e-3 m.s-1")
        self.assertEqual(q1.name, "x")
        self.assertAlmostEqual(q1.value, 1.5e-3)
        self.assertAlmostEqual(q1.sd, 2.0e-4)
        self.assertEqual(q1.units, Units("m.s-1"))

    def test_parse_units_with_space(self):
        with self.assertRaises((QuantityError, UnitsError)):
            Quantity.parse("1.5(2) m s-1")


if __name__ == "__main__":
    unittest.main()
