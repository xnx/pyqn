"""PyQn: parse, validate, manipulate and convert physical units and quantities."""

# SPDX-License-Identifier: GPL-3.0-or-later

from . import domains
from .dimensions import Dimensions
from .domains import Domain
from .quantity import Quantity, QuantityError
from .units import Units, UnitsConversionError, UnitsError, convert

__version__ = "2.1.0.dev0"

__all__ = [
    "Dimensions",
    "Domain",
    "Quantity",
    "QuantityError",
    "Units",
    "UnitsConversionError",
    "UnitsError",
    "convert",
    "domains",
]
