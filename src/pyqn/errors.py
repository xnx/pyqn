# errors.py
# The exception raised for errors in units, which pyqn.units and
# pyqn.atom_unit also provide.
#
# Copyright (C) 2012-2016 Christian Hill
# SPDX-License-Identifier: GPL-3.0-or-later


class UnitsError(Exception):
    """
    An Exception class for errors that might occur whilst manipulating units.

    """

    def __init__(self, error_str):
        self.error_str = error_str

    def __str__(self):
        return self.error_str
