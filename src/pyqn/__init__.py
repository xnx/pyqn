"""PyQn: parse, validate, manipulate and convert physical units and quantities."""

# SPDX-License-Identifier: GPL-3.0-or-later

from .dimensions import Dimensions
from .quantity import Quantity, QuantityError
from .units import Units, UnitsConversionError, UnitsError, convert

__version__ = "2.0.0"

__all__ = [
    "Dimensions",
    "Quantity",
    "QuantityError",
    "Units",
    "UnitsConversionError",
    "UnitsError",
    "convert",
]
