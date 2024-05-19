# dimensions.py
# A class representing the dimensions of a physical quantity's units, in
# terms of powers of length (L), mass (M), time (T), temperature (Theta),
# amount of substance (Q), current (C) and luminous intensity (I).
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


class Dimensions(object):
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
    dim_index = {}
    for i, dim_name in enumerate(dim_names):
        dim_index[dim_name] = i

    def __init__(self, dims=None, **kwargs):
        self.dims = [0] * 7
        if dims:
            # initialize by dims array
            if not kwargs:
                self.dims = dims
            else:
                raise ValueError("Bad initialisation of Dimensions object")
        else:
            # initialize by keyword arguments
            for dim_name in kwargs:
                self.dims[self.dim_index[dim_name]] = kwargs[dim_name]

    def __mul__(self, other):
        new_dims = []
        for i, dim in enumerate(self.dims):
            new_dims.append(dim + other.dims[i])
        return Dimensions(tuple(new_dims))

    def __truediv__(self, other):
        new_dims = []
        for i, dim in enumerate(self.dims):
            new_dims.append(dim - other.dims[i])
        return Dimensions(tuple(new_dims))

    def __pow__(self, pow):
        new_dims = []
        for dim in self.dims:
            new_dims.append(dim * pow)
        return Dimensions(tuple(new_dims))

    def __str__(self):
        s_dims = []
        for i, dim_name in enumerate(self.dim_names):
            if self.dims[i] != 0:
                this_s_dim = dim_name
                if self.dims[i] != 1:
                    this_s_dim += "%d" % self.dims[i]
                s_dims.append(this_s_dim)
        if len(s_dims) == 0:
            return "[dimensionless]"
        return ".".join(s_dims)

    def __repr__(self):
        return str(self.dims)

    def __eq__(self, other):
        for i, dim in enumerate(self.dims):
            if other.dims[i] != dim:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


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
