import unittest
from pyqn.units import Units


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
