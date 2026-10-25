"""Unit tests for the immutability, hashing and copying of the core objects."""

import copy
import pickle
import unittest
from dataclasses import FrozenInstanceError

from pyqn.base_unit import base_unit_stems
from pyqn.dimensions import Dimensions
from pyqn.quantity import Quantity
from pyqn.si import si_prefixes
from pyqn.units import Units


class ImmutabilityCheck(unittest.TestCase):
    """Check that the core objects cannot be changed."""

    def test_attributes_cannot_be_set(self):
        u1 = Units("kg.m-3")
        for obj, attr in (
            (u1, "atom_units"),
            (u1, "dims"),
            (u1, "undef"),
            (u1.atom_units[0], "exponent"),
            (u1.atom_units[0], "prefix"),
            (u1.dims, "dims"),
            (base_unit_stems["m"], "fac"),
            (si_prefixes["k"], "power"),
        ):
            with self.assertRaises(FrozenInstanceError, msg=f"{obj!r}.{attr}"):
                setattr(obj, attr, None)

    def test_atom_units_is_a_tuple(self):
        self.assertIsInstance(Units("kg.m-3").atom_units, tuple)
        self.assertEqual(Units([]).atom_units, ())
        self.assertEqual(Units("undef").atom_units, ())


class HashingCheck(unittest.TestCase):
    """Check that equal objects hash equal, so they work in sets and dicts."""

    def test_equal_units_hash_equal(self):
        for s1, s2 in (
            ("J", "kg.m2.s-2"),
            ("cm.cm", "cm2"),
            ("Hz", "s-1"),
            ("m/m", "1"),
        ):
            self.assertEqual(Units(s1), Units(s2))
            self.assertEqual(hash(Units(s1)), hash(Units(s2)), msg=(s1, s2))
        self.assertEqual(Units("1"), 1)
        self.assertEqual(hash(Units("1")), hash(1))
        self.assertEqual(hash(Dimensions(L=1)), hash(Units("m").dims))

    def test_units_in_sets_and_dicts(self):
        units = {Units("J"), Units("kg.m2.s-2"), Units("eV"), Units("m")}
        self.assertEqual(len(units), 3)
        names = {Units("J"): "joule", Units("m"): "metre"}
        self.assertEqual(names[Units("kg.m2.s-2")], "joule")
        self.assertIn(Units("1"), {1})
        self.assertNotIn(Units("deg"), {1})

    def test_undef_hash(self):
        # like NaN, undefined units are equal to nothing, not even themselves
        u1 = Units("undef")
        self.assertEqual(len({u1, Units("undef")}), 2)
        self.assertIn(u1, {u1})


class CopyingCheck(unittest.TestCase):
    """Check that the core objects can be copied and pickled."""

    def test_copy_and_pickle(self):
        u1 = Units("kg1/2.m-3/2")
        q1 = Quantity(name="x", value=1.5, units="kJ.mol-1", sd=0.1)
        for copier in (
            copy.copy,
            copy.deepcopy,
            lambda obj: pickle.loads(pickle.dumps(obj)),
        ):
            self.assertEqual(copier(u1), u1)
            self.assertEqual(str(copier(u1)), str(u1))
            self.assertEqual(copier(u1.dims), u1.dims)
            self.assertTrue(copier(Units("undef")).undef)
            self.assertEqual(str(copier(q1)), str(q1))


if __name__ == "__main__":
    unittest.main()
