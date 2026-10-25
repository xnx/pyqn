"""Check that the tables of units and domains in README.rst are up to date."""

import unittest
from pathlib import Path

from pyqn.list_base_units import list_domains_rst, list_units_rst

readme = Path(__file__).resolve().parent.parent / "README.rst"


class ReadmeCheck(unittest.TestCase):
    def table(self, marker):
        s_readme = readme.read_text(encoding="utf-8")
        start, end = f".. {marker}-table-start\n", f".. {marker}-table-end\n"
        return s_readme[s_readme.index(start) + len(start) : s_readme.index(end)]

    def test_units_table(self):
        self.assertEqual(
            self.table("units").strip(),
            list_units_rst(),
            "README.rst is out of date: see python -m pyqn.list_base_units rst",
        )

    def test_domains_table(self):
        self.assertEqual(
            self.table("domains").strip(),
            list_domains_rst(),
            "README.rst is out of date: see python -m pyqn.list_base_units domains",
        )


if __name__ == "__main__":
    unittest.main()
