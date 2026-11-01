"""Unit tests for the immutability, hashing and copying of the core objects."""

import base64
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


# A pickle made with pyqn 2.0.1 of {"molar": Units("kJ.mol-1"), "rational":
# Units("m-3.Pa-1/2"), "undef": Units("undef"), "quantity": Quantity(name="x",
# value=1.5, units="kg.m-3", sd=0.1)}.
pickle_from_2_0_1 = base64.b64decode(
    "gASVTQUAAAAAAAB9lCiMBW1vbGFylIwKcHlxbi51bml0c5SMBVVuaXRzlJOUKYGUfZQojAV1"
    "bmRlZpSJjARkaW1zlIwPcHlxbi5kaW1lbnNpb25zlIwKRGltZW5zaW9uc5STlCmBlH2UaAgo"
    "SwJLAUr+////SwBK/////0sASwB0lHNijAphdG9tX3VuaXRzlF2UKIwOcHlxbi5hdG9tX3Vu"
    "aXSUjAhBdG9tVW5pdJSTlCmBlH2UKIwJYmFzZV91bml0lIwOcHlxbi5iYXNlX3VuaXSUjAhC"
    "YXNlVW5pdJSTlCmBlH2UKIwEc3RlbZSMAUqUjARuYW1llIwFam91bGWUjAl1bml0X3R5cGWU"
    "jAZlbmVyZ3mUjANmYWOURz/wAAAAAAAAjAtkZXNjcmlwdGlvbpSMAJSMBWxhdGV4lGgdaAho"
    "CymBlH2UaAhdlChLAksBSv7///9LAEsASwBLAGVzYnVijAhleHBvbmVudJRLAYwGcHJlZml4"
    "lIwBa5SMCXNpX3ByZWZpeJSMB3B5cW4uc2mUjAhTSVByZWZpeJSTlCmBlH2UKGgqaCtoHowE"
    "a2lsb5SMBXBvd2VylEsDaCJN6AN1YnViaBMpgZR9lChoFmgZKYGUfZQoaByMA21vbJRoHowE"
    "bW9sZZRoIIwGYW1vdW50lGgiRz/wAAAAAAAAaCNoJGglaDhoCGgLKYGUfZRoCF2UKEsASwBL"
    "AEsASwFLAEsAZXNidWJoKUr/////aCpOdWJldWKMCHJhdGlvbmFslGgEKYGUfZQoaAeJaAho"
    "CymBlH2UaAgojAlmcmFjdGlvbnOUjAhGcmFjdGlvbpSTlIwELTUvMpSFlFKUaEWMBC0xLzKU"
    "hZRSlGhFjAExlIWUUpRoRYwBMJSFlFKUaEWMATCUhZRSlGhFjAEwlIWUUpRoRYwBMJSFlFKU"
    "dJRzYmgPXZQoaBMpgZR9lChoFmgZKYGUfZQoaByMAW2UaB6MBW1ldHJllGggjAZsZW5ndGiU"
    "aCJHP/AAAAAAAABoI2gkaCVoYWgIaAspgZR9lGgIXZQoSwFLAEsASwBLAEsASwBlc2J1Ymgp"
    "Sv3///9oKk51YmgTKYGUfZQoaBZoGSmBlH2UKGgcjAJQYZRoHowGcGFzY2FslGggjAhwcmVz"
    "c3VyZZRoIkc/8AAAAAAAAGgjaCRoJWhraAhoCymBlH2UaAgoSv////9LAUr+////SwBLAEsA"
    "SwB0lHNidWJoKWhFjAQtMS8ylIWUUpRoKk51YmV1YmgHaAQpgZR9lChoB4hoCE51YowIcXVh"
    "bnRpdHmUjA1weXFuLnF1YW50aXR5lIwIUXVhbnRpdHmUk5QpgZR9lChoHowBeJRoJWh8jARo"
    "dG1slGh8jApkZWZpbml0aW9ulE6MBXZhbHVllEc/+AAAAAAAAIwCc2SURz+5mZmZmZmajAV1"
    "bml0c5RoBCmBlH2UKGgHiWgIaAspgZR9lGgIKEr9////SwFLAEsASwBLAEsAdJRzYmgPXZQo"
    "aBMpgZR9lChoFmgZKYGUfZQoaByMAWeUaB6MBGdyYW2UaCCMBG1hc3OUaCJHP1BiTdLxqfxo"
    "I2gkaCVojGgIaAspgZR9lGgIXZQoSwBLAUsASwBLAEsASwBlc2J1YmgpSwFoKmgraCxoLymB"
    "lH2UKGgqaCtoHmgyaDNLA2giTegDdWJ1YmgTKYGUfZQoaBZoGSmBlH2UKGgcaGFoHmhiaCBo"
    "Y2giRz/wAAAAAAAAaCNoJGglaGFoCGgLKYGUfZRoCF2UKEsBSwBLAEsASwBLAEsAZXNidWJo"
    "KUr9////aCpOdWJldWJ1YnUu"
)


class OldPicklesCheck(unittest.TestCase):
    """Check that pickles from before the core objects were frozen are read."""

    def setUp(self):
        self.objects = pickle.loads(pickle_from_2_0_1)

    def test_units(self):
        for name, s_units in (("molar", "kJ.mol-1"), ("rational", "m-3.Pa-1/2")):
            u1 = self.objects[name]
            self.assertIsInstance(u1.atom_units, tuple)
            self.assertIs(u1.undef, False)
            self.assertEqual(u1, Units(s_units))
            self.assertEqual(str(u1), s_units)
            self.assertEqual(u1.dims, Units(s_units).dims)
            self.assertEqual(hash(u1), hash(Units(s_units)))
            self.assertEqual(u1.to_si(), Units(s_units).to_si())
            self.assertEqual((u1 * u1) / u1, Units(s_units))

    def test_undef(self):
        u1 = self.objects["undef"]
        self.assertIs(u1.undef, True)
        self.assertEqual(u1.atom_units, ())
        self.assertEqual(str(u1), "undef")

    def test_quantity(self):
        q1 = self.objects["quantity"]
        self.assertEqual(str(q1), "x = 1.5 kg.m-3")
        self.assertEqual(q1.sd, 0.1)
        self.assertEqual(q1.units, Units("kg.m-3"))

    def test_objects_round_trip(self):
        # what is read from an old pickle can be pickled again
        u1 = pickle.loads(pickle.dumps(self.objects["molar"]))
        self.assertEqual(u1, Units("kJ.mol-1"))


if __name__ == "__main__":
    unittest.main()
