import numpy as np
from .units import Units
from .quantity import Quantity

class qnArrayTwoError(Exception):
    def __init__(self, error_str):
        self.error_str = error_str
    def __str__(self):
        return self.error_str

class qnArrayTwo(np.ndarray):
    def __new__(cls, input_array, info=None, units=None, sd=None):
        """ Initialises a qnArray as a child class derived from the 
        numpy ndarray. 
        """
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        
        if type(units) is str:
            obj.units = Units(units) #records units as Unit class
        elif type(units) is Units:
            obj.units = units
        
        # checks for length of standard deviation array
        if sd is not None:
            if len(sd) != len(input_array):
                raise qnArrayTwoError("Standard deviation array must be of the same length as values array")
            obj.sd = sd
            
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    #def __mul__(self, other):
    #    if type(other) is not Quantity:
    #        raise qnArrayTwoError("Multiplication operation can be done only using Quantity objects")
        
    def __add__(self, other):
        """ Function for adding a Quantity value to all values in 
        qnArrayTwo or adding another anArrayTwo to the current array
        """
        if type(other) is qnArrayTwo:
            if len(other) != len(self):
                raise qnArrayTwoError("Inconsistent array lengths")
            v = []
            for i in range(len(self)):
                temp_q = Quantity(value = self[i], units = self.units) + Quantity(value = other[i], units = other.units)
                v.append(temp_q.value)
            return qnArrayTwo(v, units = temp_q.units)
        if type(other) is Quantity:
            v = []
            for i in range(len(self)):
                temp_q = Quantity(value=self[i], units=self.units) + other
                v.append(temp_q.value)
            return qnArrayTwo(v, units = temp_q.units)
        else:
            raise qnArrayTwoError("Can only add two qnArray objects or a qnArray with Quantity")

    def __sub__(self, other):
        if type(other) is qnArrayTwo:
            if len(other) != len(self):
                raise qnArrayTwoError("Inconsistent array lengths")
            v = []
            for i in range(len(self)):
                temp_q = Quantity(value = self[i], units = self.units) - Quantity(value = other[i], units = other.units)
                v.append(temp_q.value)
            return qnArrayTwo(v, units = temp_q.units)

        if type(other) is Quantity:
            v = []
            for i in range(len(self)):
                temp_q = Quantity(value=self[i], units=self.units) - other
                v.append(temp_q.value)
            return qnArrayTwo(v, units = temp_q.units)
        else:
            raise qnArrayTwoError("Can only subtract two qnArray objects or a qnArray with Quantity")

    def __mul__(self, other):
        if type(other) is qnArrayTwo:
            if len(other) != len(self):
                raise qnArrayTwoError("Inconsistent array lengths")
            v = []
            for i in range(len(self)):
                temp_q = Quantity(value = self[i], units = self.units) * Quantity(value = other[i], units = other.units)
                v.append(temp_q.value)
            return qnArrayTwo(v, units = temp_q.units)

        if type(other) is Quantity:
            v = []
            for i in range(len(self)):
                temp_q = Quantity(value=self[i], units=self.units) * other
                v.append(temp_q.value)
            return qnArrayTwo(v, units = temp_q.units)
        else:
            raise qnArrayTwoError("Can only multiply two qnArray objects or a qnArray with Quantity")

    def __truediv__(self, other):
        if type(other) is qnArrayTwo:
            if len(other) != len(self):
                raise qnArrayTwoError("Inconsistent array lengths")
            v = []
            for i in range(len(self)):
                temp_q = Quantity(value = self[i], units = self.units) / Quantity(value = other[i], units = other.units)
                v.append(temp_q.value)
            return qnArrayTwo(v, units = temp_q.units)

        if type(other) is Quantity:
            v = []
            for i in range(len(self)):
                temp_q = Quantity(value=self[i], units=self.units) / other
                v.append(temp_q.value)
            return qnArrayTwo(v, units = temp_q.units)
        else:
            raise qnArrayTwoError("Can onlydividde two qnArray objects or a qnArray with Quantity")
