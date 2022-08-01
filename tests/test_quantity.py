import unittest
from pyqn.quantity import Quantity, QuantityError
from pyqn.dimensions import Dimensions, d_energy
from pyqn.units import UnitsError


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
        # q1 = Quantity(value = 200, units = 'J')
        # q2 = q1.convert_units_to('eV')
        # self.assertAlmostEqual(q2.value,1.2483019242e+21,places=2)
        pass


if __name__ == "__main__":
    unittest.main()
