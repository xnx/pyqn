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
        self.nparr = np.array([])
        self.units = units
        if len(values) is not 0:
            if type(values) is list or type(values) is np.ndarray:
                if type(values[0]) is not str:
                    self.values = values
                else:
                    raise qnArrayError("Values must be numbers")
            else:
                raise qnArrayError("Values only an array/list of values")
        for v in values:
            self.nparr = np.append(self.nparr, Quantity(value=v, units = units))
            
    def __mul__(self, other):
        return self.nparr * other
    def __truediv__(self, other):
        return self.nparr / other
