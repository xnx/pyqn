# -*- coding: utf-8 -*-
# symbol.py

# Christian Hill, 29/3/13
# Department of Physics and Astronomy, University College London
# christian.hill@ucl.ac.uk

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
