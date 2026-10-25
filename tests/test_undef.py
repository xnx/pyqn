import unittest

from pyqn.quantity import Quantity
from pyqn.units import Units, UnitsError


class TestUndef(unittest.TestCase):
    def test_undef(self):
        u1 = Units("undef")
        self.assertTrue(u1.undef)
        self.assertEqual(repr(u1), "undef")
        self.assertEqual(u1.html, "undef")

        u2 = Units("cm.hr-1/m")

        for u3 in (Units("undef"), 1, None):
            self.assertFalse(u1 == u3)
            self.assertTrue(u1 != u3)
            self.assertFalse(u2 == u3)
            self.assertTrue(u2 != u3)

    def test_undef_arithmetic(self):
        u1 = Units("undef")
        u2 = Units("m")
        for u3 in (u1 * u2, u2 * u1, u1 / u2, u2 / u1, u1**2, 1 / u1, "s" * u1):
            self.assertTrue(u3.undef, msg=str(u3))
        self.assertFalse((u1**0).has_units())
        self.assertTrue(u1.has_units())

    def test_undef_conversion(self):
        u1 = Units("undef")
        for convert in (
            lambda: u1.conversion("m"),
            lambda: Units("m").conversion(u1),
            u1.to_si,
            u1.get_dims,
        ):
            with self.assertRaises(UnitsError):
                convert()

    def test_undef_quantities(self):
        q1 = Quantity(value=2, units="undef")
        q2 = Quantity(value=3, units="s", sd=0.1)
        q3 = q1 * q2
        self.assertEqual(q3.value, 6)
        self.assertTrue(q3.units.undef)
        self.assertEqual(q3.as_str(), "6 undef")
        for combine in (lambda: q1 + q1, lambda: q1 - q1):
            with self.assertRaises(UnitsError):
                combine()
        with self.assertRaises(UnitsError):
            q1.convert_units_to("m")
