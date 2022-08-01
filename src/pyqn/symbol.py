# symbol.py
# A class representing a symbol (perhaps the label for a physical quantity
# represented as a Quantity object) with a name in text,  LaTeX and HTML.
#
# Copyright (C) 2012-2016 Christian Hill
# Department of Physics and Astronomy, University College London
# christian.hill@ucl.ac.uk
#
# This file is part of PyQn
#
# PyQn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyQn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyQn.  If not, see <http://www.gnu.org/licenses/>


class Symbol(object):
    """
    A Python class representing a symbol - typically the label for a physical
    quantity - with plain-text (utf8-encoded string), LaTeX, and HTML versions
    and perhaps a definition or description.

    """

    def __init__(self, name=None, latex=None, html=None, definition=None):
        """
        Initialize the Symbol object: if they are not given, the LaTeX and HTML
        descriptions of the Symbol are set to its plain-text name.

        """

        self.name = name
        self.latex = latex
        if not latex:
            self.latex = name
        self.html = html
        if not html:
            self.html = name
        self.definition = definition

    def __str__(self):
        """
        Return the string representation of the Symbol, its name attribute.
        """
        return self.name
