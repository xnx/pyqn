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
        if sd is None:
            obj.sd = np.zeros(len(input_array))
        else:
            obj.sd = np.array(sd)
        
        if type(units) is str:
            obj.units = Units(units) #records units as Unit class
        elif type(units) is Units:
            obj.units = units            
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)
        self.sd = getattr(obj, 'sd', None)
        
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in ufunc_dict:
            alg_func = ufunc_dict[ufunc]
            # check for units matching
            if hasattr(inputs[0],'units') and hasattr(inputs[1],'units'):
                if inputs[0].units != inputs[1].units:
                    raise qnArrayTwoError('Units must match')
            # if both are qn arrays
            if (type(inputs[1]) is qnArrayTwo) and (type(inputs[0]) is qnArrayTwo):
                return qnArrayTwo(getattr(np.asarray(inputs[0]), alg_func)(np.asarray(inputs[1])), 
                                  units = inputs[0].units, 
                                  sd = np.sqrt(inputs[0].sd**2+inputs[1].sd**2))
            # if one input is a quantity
            if type(inputs[1]) is Quantity:
                return qnArrayTwo(getattr(np.asarray(inputs[0]), alg_func)(inputs[1].value),
                                  units = inputs[0].units,
                                  sd = np.sqrt(inputs[0].sd**2+inputs[1].sd**2))
            if type(inputs[0]) is Quantity:
                return qnArrayTwo(getattr(inputs[0].value, alg_func)(np.asarray(inputs[1])),
                                  units = inputs[1].units,
                                  sd = np.sqrt(inputs[1].sd**2+inputs[0].sd**2))
            # for all other object types
            if type(inputs[0]) is qnArrayTwo:
                return qnArrayTwo(getattr(np.asarray(inputs[0]), alg_func)(inputs[1]),
                                  units = inputs[0].units,
                                  sd = inputs[0].sd)
            else:
                return qnArrayTwo(getattr(inputs[0], alg_func)(np.asarray(inputs[1])),
                                  units = inputs[1].units,
                                  sd = inputs[1].sd)

    def __eq__(self, other):
        if all(super(qnArrayTwo, self).__eq__(super(qnArrayTwo, other))) and (self.units == other.units):
            return True
        else:
            return False

    #def __neq__(self, other):
    #    return all(not self.__eq__(other))

    @property
    def html_str(self):
        html_chunks = []
        for i in range(len(self)):
            html_chunks.append('{} {}'.format(self[i],self.units))
        return ', '.join(html_chunks)
        
ufunc_dict = {  np.add: '__add__',
                np.subtract: '__sub__',
                np.multiply: '__mul__',
                np.divide: '__truediv__'}
