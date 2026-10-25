# list_base_units.py
# List the base units, by group, as text, HTML or reStructuredText tables:
#     python -m pyqn.list_base_units <text|html|rst>
# The rst output is the table of units in README.rst.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys

from .base_unit import base_units, metric, metric_and_hecto, no_prefixes
from .si import si_prefixes


def describe_prefixes(base_unit):
    """Return a description of the SI prefixes that base_unit takes."""
    named = {metric_and_hecto: "all", metric: "all except h", no_prefixes: "none"}
    if base_unit.prefixes in named:
        return named[base_unit.prefixes]
    return ", ".join(p for p in si_prefixes if p in base_unit.prefixes)


def list_units_text():
    lines = []
    for group_name, base_unit_group in base_units:
        lines += ["", group_name, "=" * len(group_name)]
        lines.append(f"{'Unit':6s} {'Name':30s} {'Dimensions':20s} Prefixes")
        lines.append(f"{'----':6s} {'----':30s} {'----------':20s} --------")
        for base_unit in base_unit_group:
            lines.append(
                f"{base_unit.stem:6s} {base_unit.name:30s} {str(base_unit.dims):20s}"
                f" {describe_prefixes(base_unit)}"
            )
    return "\n".join(lines)


def list_units_html():
    lines = []
    for group_name, base_unit_group in base_units:
        lines.append(f"<h4>{group_name}</h4>")
        lines.append('<table class="my-table">')
        lines.append(
            "<tr><th>Unit</th><th>Name</th><th>Dimensions</th><th>Prefixes</th></tr>"
        )
        for base_unit in base_unit_group:
            lines.append(
                f"<tr><td>{base_unit.stem}</td><td>{base_unit.name}</td>"
                f"<td>{base_unit.dims}</td><td>{describe_prefixes(base_unit)}</td></tr>"
            )
        lines.append("</table>")
    return "\n".join(lines)


def list_units_rst():
    blocks = []
    for group_name, base_unit_group in base_units:
        rows = [("Unit", "Name", "Dimensions", "Prefixes")]
        for base_unit in base_unit_group:
            rows.append(
                (
                    f"``{base_unit.stem}``",
                    base_unit.name,
                    str(base_unit.dims),
                    describe_prefixes(base_unit),
                )
            )
        widths = [max(len(row[i]) for row in rows) for i in range(4)]
        rule = "  ".join("=" * width for width in widths)
        lines = [f"**{group_name}**", "", rule]
        for i, row in enumerate(rows):
            lines.append("  ".join(f"{s:{w}s}" for s, w in zip(row, widths)).rstrip())
            if i == 0:
                lines.append(rule)
        lines.append(rule)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


if __name__ == "__main__":
    list_units = {
        "text": list_units_text,
        "html": list_units_html,
        "rst": list_units_rst,
    }
    try:
        list_type = sys.argv[1]
        assert list_type in list_units
    except (IndexError, AssertionError):
        print(f"usage:\n{sys.argv[0]} <list_type>")
        print("where <list_type> is text, html or rst")
        sys.exit(1)

    print(list_units[list_type]())
