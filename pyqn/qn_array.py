from .symbol import Symbol
import numpy as np
from .quantity import Quantity

class qnArrayError(Exception):
    def __init__(self, error_str):
        self.error_str = error_str
    def __str__(self):
        return self.error_str

class qnArray(Symbol):
    def __init__(self, name=None, latex=None, html=None, values=None, units=None, sd=None, definition=None):
        Symbol.__init__(self, name, latex, html, definition)
        #if type(values)==list:
        #    self.values = np.array(values)
        #elif type(values)==numpy.ndarray:
        #    self.values = values
        #else:
        #    raise qnArrayError
        self.nparr = np.array([])
        self.units = units
        self.values = values
        for v in values:
            self.nparr = np.append(self.nparr, Quantity(value=v, units = units))
            
    def __mul__(self, other):
        return self.nparr * other

    def values(self):
        return self.values
    
    def units(self):
        return self.units
