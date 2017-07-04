import numpy as np
from .units import Units
from .quantity import Quantity

class qnArrayTwoError(Exception):
	def __init__(self, error_str):
        self.error_str = error_str
    def __str__(self):
        return self.error_str

class qnArrayTwo(np.ndarray):
    def __new__(cls, input_array, info=None, units=None, sd_arr=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        obj.units = Units(units)
        if sd_arr is not None:
			if len(input_array) != len(input_array):
				raise qnArrayTwoError("Standard deviation array must be of the same length as values array")
			obj.sd = sd_arr
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

	def __mul__(self, other):
		if type(other) is not Quantity:
			raise qnArrayTwoError("Multiplication operation can be done only using Quantity objects")
		
