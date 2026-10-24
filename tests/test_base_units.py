"""Check the base units' SI factors against their definitions and CODATA 2022."""

import math
import unittest

import pytest

from pyqn.base_unit import base_unit_stems

inch, lb, au = 0.0254, 0.45359237, 1.495978707e11
us_gal = 231 * inch**3


def approx(expected, rel=1.0e-12):
    """
    Compare with a relative tolerance only: pytest.approx's default absolute
    tolerance (1e-12) would accept any two very small factors.

    """
    return pytest.approx(expected, rel=rel, abs=0)


def check_factors(test_case, factors):
    for stem, fac in factors.items():
        test_case.assertEqual(base_unit_stems[stem].fac, approx(fac), msg=stem)


class BaseUnitFactorsCheck(unittest.TestCase):
    """Check the factors relating each BaseUnit to its SI unit."""

    def test_defined_units(self):
        check_factors(
            self,
            {
                "atm": 101325,
                "Torr": 101325 / 760,
                "mmHg": 133.322387415,
                "cal": 4.184,
                "inch": inch,
                "ft": 12 * inch,
                "yard": 36 * inch,
                "fur": 7920 * inch,
                "mi": 63360 * inch,
                "NM": 1852,
                "lb": lb,
                "oz": lb / 16,
                "st": 14 * lb,
                "fir": 90 * lb,
                "gal": 4.54609e-3,
                "pt": 4.54609e-3 / 8,
                "USgal": us_gal,
                "USpt": us_gal / 8,
                "deg": math.pi / 180,
                "arcmin": math.pi / 10800,
                "asec": math.pi / 648000,
                "AU": au,
                "pc": 648000 / math.pi * au,
                "yr": 31557600,
                "ly": 299792458 * 31557600,
                "D": 1.0e-21 / 299792458,
            },
        )

    def test_knot(self):
        check_factors(self, {"kn": 1852 / 3600})

    def test_codata_2022(self):
        e, h, m_u = 1.602176634e-19, 6.62607015e-34, 1.66053906892e-27
        check_factors(
            self,
            {
                "e": e,
                "eV": e,
                "hbar": h / (2 * math.pi),
                "u": m_u,
                "amu": m_u,
                "Da": m_u,
                "m_e": 9.1093837139e-31,
                "E_h": 4.359744722206e-18,
                "Ry": 2.179872361103e-18,
                "a0": 5.29177210544e-11,
            },
        )


if __name__ == "__main__":
    unittest.main()
