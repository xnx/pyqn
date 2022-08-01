# list_base_units.py
import sys
from base_unit import base_units


def list_units_text():
    for group_name, base_unit_group in base_units:
        print()
        print(group_name)
        print("=" * len(group_name))
        print("{0:6s} {1:30s} {2:20s}".format("Unit", "Name", "Dimensions"))
        print("{0:6s} {1:30s} {2:20s}".format("----", "----", "----------"))
        for base_unit in base_unit_group:
            print(
                "{0:6s} {1:30s} {2:20s}".format(
                    unicode(base_unit.stem), base_unit.name, base_unit.dims
                )
            )


def list_units_html():
    for group_name, base_unit_group in base_units:
        print("<h4>{0}</h4>".format(group_name))
        print('<table class="my-table">')
        print("<tr><th>Unit</th><th>Name</th><th>Dimensions</th></tr>")
        for base_unit in base_unit_group:
            # NB we need to encode('utf-8') to pipe in that encoding to the
            # shell, e.g. to create a file
            print(
                "<tr><td>{0:s}</td><td>{1:s}</td><td>{2:s}</td><td></tr>".format(
                    unicode(base_unit.stem), base_unit.name, base_unit.dims
                ).encode("utf-8")
            )
        print("</table>")


try:
    list_type = sys.argv[1]
    assert list_type in ("text", "html")
except (IndexError, AssertionError):
    print("usage:\n{0} <list_type>".format(sys.argv[0]))
    print("where <list_type> is text or html")
    sys.exit(1)

if list_type == "text":
    list_units_text()
else:
    list_units_html()
