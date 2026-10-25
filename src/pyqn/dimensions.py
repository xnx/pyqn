# dimensions.py
# A class representing the dimensions of a physical quantity's units, in
# terms of powers of length (L), mass (M), time (T), temperature (Theta),
# amount of substance (Q), current (C) and luminous intensity (I).
#
# Copyright (C) 2012-2016 Christian Hill
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, init=False)
class Dimensions:
    """
    The dimensions of a physical quantity's units: an immutable, hashable
    tuple of the powers of the seven base dimensions, named in dim_names,
    and the powers of any pseudo-dimensions. A pseudo-dimension, such as
    'volume fraction', is a named dimension beyond the seven of the SI, which
    cancels only against itself: it tells apart kinds of quantity that the SI
    gives the same dimensions, such as mass and volume fractions.

    """

    # these are the abbreviations for Length, Mass, Time, Temperature,
    # Quantity (amount of substance), Current, and Luminous Intensity:
    dim_names = ["L", "M", "T", "Theta", "Q", "C", "I"]
    dim_desc = [
        "length",
        "mass",
        "time",
        "temperature",
        "amount",
        "current",
        "luminous intensity",
    ]
    dim_index = {dim_name: i for i, dim_name in enumerate(dim_names)}

    dims: tuple
    # the non-zero powers of any pseudo-dimensions, as a tuple of
    # (name, power) pairs sorted by name
    pseudo: tuple

    def __init__(self, dims=None, pseudo=None, **kwargs):
        """
        Initialize the Dimensions from a sequence of the powers of the seven
        base dimensions, or from keyword arguments such as L=1, T=-1, and
        from a dict of the powers of any pseudo-dimensions.

        """

        if dims:
            # initialize by dims array
            if kwargs or len(dims) != 7:
                raise ValueError("Bad initialisation of Dimensions object")
        else:
            # initialize by keyword arguments
            dims = [0] * 7
            for dim_name in kwargs:
                dims[self.dim_index[dim_name]] = kwargs[dim_name]
        object.__setattr__(self, "dims", tuple(dims))
        pseudo = sorted((name, power) for name, power in dict(pseudo or {}).items())
        object.__setattr__(self, "pseudo", tuple((n, p) for n, p in pseudo if p != 0))

    def _pseudo_sum(self, other, sign):
        pseudo = dict(self.pseudo)
        for name, power in other.pseudo:
            pseudo[name] = pseudo.get(name, 0) + sign * power
        return pseudo

    def __mul__(self, other):
        new_dims = []
        for i, dim in enumerate(self.dims):
            new_dims.append(dim + other.dims[i])
        return Dimensions(tuple(new_dims), self._pseudo_sum(other, 1))

    def __truediv__(self, other):
        new_dims = []
        for i, dim in enumerate(self.dims):
            new_dims.append(dim - other.dims[i])
        return Dimensions(tuple(new_dims), self._pseudo_sum(other, -1))

    def __pow__(self, pow):
        new_dims = []
        for dim in self.dims:
            new_dims.append(dim * pow)
        pseudo = {name: power * pow for name, power in self.pseudo}
        return Dimensions(tuple(new_dims), pseudo)

    def __str__(self):
        s_dims = []
        for i, dim_name in enumerate(self.dim_names):
            if self.dims[i] != 0:
                this_s_dim = dim_name
                if self.dims[i] != 1:
                    this_s_dim += str(self.dims[i])
                s_dims.append(this_s_dim)
        for name, power in self.pseudo:
            s_dims.append(f"[{name}]" + ("" if power == 1 else str(power)))
        if len(s_dims) == 0:
            return "[dimensionless]"
        return ".".join(s_dims)

    def __repr__(self):
        if self.pseudo:
            return f"{self.dims} {dict(self.pseudo)}"
        return str(self.dims)


d_dimensionless = Dimensions()
d_quantity = Dimensions(Q=1)
d_frequency = Dimensions(T=-1)
d_length = Dimensions(L=1)
d_area = d_length**2
d_volume = d_length**3
d_time = Dimensions(T=1)
d_mass = Dimensions(M=1)
d_energy = Dimensions(M=1, L=2, T=-2)
d_force = d_mass * d_length / d_time**2
d_pressure = d_force / d_area
d_current = Dimensions(C=1)
d_charge = d_current * Dimensions(T=1)
d_voltage = d_energy / d_charge  # 1 V = 1 J/C
d_magfield_strength = d_voltage * d_time / d_area  # 1 T = 1 V.s/m^2
d_magnetic_flux = d_voltage * d_time  # 1 Wb = 1 V.s
d_temperature = Dimensions(Theta=1)

# Pseudo-dimensions for the kinds of fraction, which the SI gives dimension 1
d_mass_fraction = Dimensions(pseudo={"mass fraction": 1})
d_volume_fraction = Dimensions(pseudo={"volume fraction": 1})
d_amount_fraction = Dimensions(pseudo={"amount fraction": 1})
