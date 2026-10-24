"""Check that the table of units in README.rst is up to date."""

import unittest
from pathlib import Path

from pyqn.list_base_units import list_units_rst

readme = Path(__file__).resolve().parent.parent / "README.rst"


class ReadmeCheck(unittest.TestCase):
    def test_units_table(self):
        s_readme = readme.read_text(encoding="utf-8")
        start, end = ".. units-table-start\n", ".. units-table-end\n"
        table = s_readme[s_readme.index(start) + len(start) : s_readme.index(end)]
        self.assertEqual(
            table.strip(),
            list_units_rst(),
            "README.rst is out of date: see python -m pyqn.list_base_units rst",
        )


if __name__ == "__main__":
    unittest.main()
