import numpy as np
from .units import Units
from .quantity import Quantity

class qnArrayTwoError(Exception):
    def __init__(self, error_str):
        self.error_str = error_str
    def __str__(self):
        return self.error_str

class qnArrayTwo(np.ndarray):
    def __new__(cls, input_array, info=None, units='1', sd=None):
        """ Initialises a qnArray as a child class derived from the 
        numpy ndarray. 
        """
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        
        if type(units) is str:
            obj.units = Units(units) #records units as Unit class
        elif type(units) is Units:
            obj.units = units            
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    def __add__(self, other):
        """ Function for adding a Quantity value to all values in 
        qnArrayTwo or adding another anArrayTwo to the current array
        """
        if type(other) is qnArrayTwo:
            if self.units != other.units:
                raise qnArrayTwoError('Units of the two array must be compatible')
            return qnArrayTwo(super(qnArrayTwo, self).__add__(super(qnArrayTwo, other)), units = self.units)
        if type(other) is Quantity:
            if self.units != other.units:
                raise qnArrayTwoError('Units of the two array must be compatible')
            return qnArrayTwo(super(qnArrayTwo, self).__add__(other.value), units = self.units)
        return qnArrayTwo(super(qnArrayTwo, self).__add__(other), units = self.units)
        
    def __sub__(self, other):
        if type(other) is qnArrayTwo:
            if self.units != other.units:
                raise qnArrayTwoError('Units of the two array must be compatible')
            return qnArrayTwo(super(qnArrayTwo, self).__sub__(super(qnArrayTwo, other)), units = self.units)
        if type(other) is Quantity:
            if self.units != other.units:
                raise qnArrayTwoError('Units of the two array must be compatible')
            return qnArrayTwo(super(qnArrayTwo, self).__sub__(other.value), units = self.units)
        return qnArrayTwo(super(qnArrayTwo, self).__sub__(other), units = self.units)
            
    def __mul__(self, other):
        if type(other) is qnArrayTwo:
            return qnArrayTwo(super(qnArrayTwo, self).__mul__(super(qnArrayTwo, other)), units = self.units * other.units)
        if type(other) is Quantity:
            return qnArrayTwo(super(qnArrayTwo, self).__mul__(other.value), units = self.units * other.units)
        return qnArrayTwo(super(qnArrayTwo, self).__mul__(other), units = self.units)
            
    def __truediv__(self, other):
        if type(other) is qnArrayTwo:
            return qnArrayTwo(super(qnArrayTwo, self).__truediv__(super(qnArrayTwo, other)), units = self.units / other.units)
        if type(other) is Quantity:
            return qnArrayTwo(super(qnArrayTwo, self).__truediv__(other.value), units = self.units / other.units)
        return qnArrayTwo(super(qnArrayTwo, self).__truediv__(other), units = self.units)
        
    def __pow__(self, power):
        return qnArrayTwo(super(qnArrayTwo, self).__pow__(power), units = self.units ** power)

    #def __eq__(self, other):
    #    if all(self == other) and (self.units == other.units) and (self.sd == other.sd):
    #        return True
    #    else:
    #        return False

    @property
    def html_str(self):
        html_chunks = []
        for i in range(len(self)):
            html_chunks.append(Quantity(value=self[i],units = self.units, sd = self.sd[i]).html_str)
        return ', '.join(html_chunks)
