from .symbol import Symbol
import numpy as np
from .quantity import Quantity
from .units import Units

class qnArrayError(Exception):
    def __init__(self, error_str):
        self.error_str = error_str
    def __str__(self):
        return self.error_str

class qnArray(Symbol):
    def __init__(self, name=None, latex=None, html=None, values=None, 
                        units=None, sd=None, definition=None):
        Symbol.__init__(self, name, latex, html, definition)
        
        #validates units
        if type(units) is str:
            self.units = Units(units)
        elif type(units) is Units:
            self.units = units
        else:
            raise qnArrayError("Units accepted only in str or Units class form")
        
        #validates values array
        if len(values) is not 0:
            if type(values) is list or type(values) is np.ndarray:
                if type(values[0]) is not str:
                    self.values = values
                else:
                    raise qnArrayError("Values must be numbers")
            else:
                raise qnArrayError("Values only an array/list of values")
                
        if type(values) is list:
            self.nparr = np.array(values)
        elif type(values) is np.ndarray:
            self.nparr = values
            
    def __mul__(self, other):
        if type(other) is not Quantity:
            raise qnArrayError("qnArrays can obly be multiplied by Quantity objects")
        return qnArray(values = self.nparr*other.value, units = self.units*other.units)
        
    def __truediv__(self, other):
        return self.nparr / other
        
    @property
    def units_str(self):
        return str(self.units)
