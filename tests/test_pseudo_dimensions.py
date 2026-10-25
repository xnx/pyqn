"""Unit tests for pseudo-dimensions, kinds of fraction, and the molecule."""

import unittest

import pytest

from pyqn.dimensions import (
    Dimensions,
    d_mass_fraction,
    d_volume_fraction,
)
from pyqn.quantity import Quantity
from pyqn.units import Units, UnitsError

NA = 6.02214076e23


def approx(expected, rel=1.0e-9):
    """
    Compare with a relative tolerance only: pytest.approx's default absolute
    tolerance (1e-12) would accept any two very small factors.

    """
    return pytest.approx(expected, rel=rel, abs=0)


class PseudoDimensionsCheck(unittest.TestCase):
    """Unit tests for pseudo-dimensions in Dimensions objects."""

    def test_pseudo_dimensions(self):
        v = d_volume_fraction
        self.assertEqual(v.dims, (0,) * 7)
        self.assertEqual(v.pseudo, (("volume fraction", 1),))
        self.assertEqual(str(v), "[volume fraction]")
        self.assertEqual(str(v**2 / Dimensions(L=1)), "L-1.[volume fraction]2")
        # a pseudo-dimension cancels only against itself
        self.assertEqual(v / v, Dimensions())
        self.assertEqual(hash(v / v), hash(Dimensions()))
        self.assertNotEqual(v, d_mass_fraction)
        self.assertNotEqual(v, Dimensions())
        self.assertEqual(
            Dimensions(L=1, pseudo={"volume fraction": 1, "mass fraction": 0}),
            Dimensions(L=1) * v,
        )


class FractionsCheck(unittest.TestCase):
    """Unit tests for fractions of a particular kind."""

    def test_same_kind(self):
        self.assertEqual(Units("ppmv").conversion("ppbv"), approx(1.0e3))
        self.assertEqual(Units("vol%").conversion("pptv"), approx(1.0e10))
        self.assertEqual(Units("wt%").conversion("ppmw"), approx(1.0e4))
        self.assertEqual(Units("ppmv/ppmv"), 1)

    def test_different_kinds(self):
        for from_units, to_units in (
            ("ppmv", "ppmw"),
            ("ppmv", "ppm"),
            ("ppmv", "μmol/mol"),
            ("wt%", "%"),
            ("mol%", "vol%"),
        ):
            with self.assertRaises(UnitsError, msg=(from_units, to_units)):
                Units(from_units).conversion(to_units)
        self.assertNotEqual(Units("ppmv"), Units("ppm"))
        self.assertNotEqual(Units("ppmv"), Units("ppmw"))
        with self.assertRaises(UnitsError):
            Quantity(value=1, units="ppmv") + Quantity(value=1, units="ppmw")

    def test_fraction_conventions(self):
        # a convention treats its pseudo-dimension as dimensionless
        self.assertEqual(
            Units("ppmv").conversion("μmol/mol", force="volume fraction"),
            approx(1),
        )
        self.assertEqual(
            Units("wt%").conversion("1", force="mass fraction"), approx(0.01)
        )
        self.assertEqual(
            Units("mol%").conversion("%", force="amount fraction"), approx(1)
        )
        # forcing one kind does not convert another
        with self.assertRaises(UnitsError):
            Units("ppmv").conversion("ppmw", force="volume fraction")
        # but forcing both does
        self.assertEqual(
            Units("ppmv").conversion(
                "ppmw", force=("volume fraction", "mass fraction")
            ),
            approx(1),
        )
        # with other conventions too, in either order
        for force in (("volume fraction", "spec"), ("spec", "volume fraction")):
            self.assertEqual(
                Units("ppmv.cm-1").conversion("MHz", force=force),
                approx(1.0e-6 * 299792458 * 100 / 1.0e6),
                msg=force,
            )


class MoleculeCheck(unittest.TestCase):
    """Unit tests for the molecule, a unit of amount."""

    def test_molecule(self):
        self.assertEqual(Units("molec").to_si(), approx(1 / NA))
        self.assertEqual(Units("molec").dims, Dimensions(Q=1))
        # a HITRAN line intensity
        s_units = Units("cm-1/molec.cm-2")
        self.assertEqual(s_units, Units("cm.molec-1"))
        self.assertEqual(s_units.conversion("cm.mol-1"), approx(NA))
        self.assertEqual(s_units.conversion("cm", force="mol"), approx(1))
        self.assertEqual(Units("molec.cm-3").conversion("mol.L-1"), approx(1.0e3 / NA))
        self.assertEqual(Units("molec").conversion("1", force="mol"), approx(1))


if __name__ == "__main__":
    unittest.main()
