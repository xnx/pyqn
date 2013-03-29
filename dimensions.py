# -*- coding: utf-8 -*-

# Christian Hill
# v0.2 21/11/2012
# v0.1 28/11/2011
#
# The Dimensions class, representing the dimensions of a physical quantity.

class Dimensions(object):
    # these are the abbreviations for Length, Mass, Time, Temperature,
    # Quantity (amount of substance), Current, and Luminous Intensity:
    dim_names = ['L', 'M', 'T', 'Theta', 'Q', 'C', 'I']
    dim_desc = ['length', 'mass', 'time', 'temperature', 'amount',
                'current', 'luminous intensity']
    dim_index = {}
    for i, dim_name in enumerate(dim_names):
        dim_index[dim_name] = i

    def __init__(self, dims=None, **kwargs):
        self.dims = [0]*7
        if dims:
            # initialize by dims array
            if not kwargs:
                self.dims = dims
            else:
                print 'bad initialisation of Dimensions object'
                sys.exit(1)
        else:
            # initialize by keyword arguments
            for dim_name in kwargs:
                self.dims[self.dim_index[dim_name]] = kwargs[dim_name]

    def __mul__(self, other):
        new_dims = []
        for i, dim in enumerate(self.dims):
            new_dims.append(dim + other.dims[i])
        return Dimensions(tuple(new_dims))

    def __div__(self, other):
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
                    this_s_dim += '%d' % self.dims[i]
                s_dims.append(this_s_dim)
        if len(s_dims) == 0:
            return '[dimensionless]'
        return '.'.join(s_dims)

    def __eq__(self, other):
        for i, dim in enumerate(self.dims):
            if other.dims[i] != dim:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

d_dimensionless = Dimensions()
d_length = Dimensions(L=1)
d_area = d_length**2
d_volume = d_length**3
d_time = Dimensions(T=1)
d_mass = Dimensions(M=1)
d_energy = Dimensions(M=1, L=2, T=-2)
d_force = d_mass * d_length / d_time**2
d_pressure = d_force / d_area
d_charge = Dimensions(C=1) * Dimensions(T=1)
d_voltage = d_energy / d_charge     # 1 V = 1 J/C
d_magfield_strength = d_voltage * d_time / d_area   # 1 T = 1 Vs/m^2

