# conf.py
# Configuration for the Sphinx documentation of PyQn.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pyqn

project = "PyQn"
author = "Christian Hill"
copyright = "2012–2026, Christian Hill"
release = pyqn.__version__
version = ".".join(release.split(".")[:2])

extensions = ["sphinx.ext.autodoc"]
autodoc_member_order = "bysource"
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = f"PyQn {release}"
