import numpy as np
from .units import Units
from .quantity import Quantity
from .ufunc_dictionaries import *
import matplotlib.pyplot as plt

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
        if ufunc in ufunc_dict_alg:
            alg_func = ufunc_dict_alg[ufunc][0]
            sd_func = ufunc_dict_alg[ufunc][1]
            units_func = ufunc_dict_alg[ufunc][2]
            alg_func_reverse = ufunc_dict_alg[ufunc][3]

            if all([hasattr(x, 'units') for x in inputs]):
                result_units = units_func(inputs[0].units, inputs[1].units)

            # if both are qn arrays
            if (type(inputs[1]) is qnArrayTwo) and (type(inputs[0]) is qnArrayTwo):
                result_val = getattr(np.asarray(inputs[0]), alg_func)(np.asarray(inputs[1]))
                result_sd = sd_func(result_val, np.asarray(inputs[0]),
                                                np.asarray(inputs[1]),
                                                inputs[0].sd,
                                                inputs[1].sd)

            # if one input is a quantity
            elif type(inputs[1]) is Quantity:
                result_val = getattr(np.asarray(inputs[0]), alg_func)(inputs[1].value)
                result_sd = sd_func(result_val, np.asarray(inputs[0]),
                                                inputs[1].value,
                                                inputs[0].sd,
                                                inputs[1].sd)

            elif type(inputs[0]) is Quantity:
                result_val = getattr(np.asarray(inputs[1]), alg_func_reverse)(inputs[0].value)
                result_sd = sd_func(result_val, inputs[0].value,
                                                np.asarray(inputs[1]),
                                                inputs[0].sd,
                                                inputs[1].sd)

            # for all other object types
            elif type(inputs[0]) is qnArrayTwo:
                result_val = getattr(np.asarray(inputs[0]), alg_func)(inputs[1])
                result_sd = sd_func(result_val, np.asarray(inputs[0]),
                                                inputs[1],
                                                inputs[0].sd, 0)
                result_units = units_func(inputs[0].units, Units('1'))

            else:
                result_val = getattr(np.asarray(inputs[1]), alg_func_reverse)(inputs[0])
                result_sd = sd_func(result_val, inputs[0],
                                                np.asarray(inputs[1]),
                                                0, inputs[1].sd)
                result_units = units_func(Units('1'), inputs[1].units)

            return qnArrayTwo(result_val, units = result_units, sd = result_sd)
            
        elif ufunc in ufunc_dict_one_input:
			#checks if units of input are valid
            unit_test_func = ufunc_dict_one_input[ufunc][1]
            inputs_checked = unit_test_func(inputs[0])
            
            #extracts functions for finding sd and units
            sd_func = ufunc_dict_one_input[ufunc][0]
            units_func = ufunc_dict_one_input[ufunc][2]
            
            #calculates results
            result_val = ufunc(np.asarray(inputs_checked))
            result_sd = sd_func(result_val, np.asarray(inputs_checked), np.asarray(inputs_checked.sd))
            result_units = units_func(inputs[0].units)
            return qnArrayTwo(result_val, units = result_units, sd = result_sd)

        elif ufunc in ufunc_dict_two_inputs:
            unit_test_func = ufunc_dict_two_inputs[ufunc][1]
            input1 = unit_test_func(inputs[0])
            input2 = unit_test_func(inputs[1])
            sd_func = ufunc_dict_two_inputs[ufunc][0]
            units_func = ufunc_dict_two_inputs[ufunc][2]
            result_val = ufunc(np.asarray(input1),np.asarray(input2))
            result_sd = sd_func(result_val, np.asarray(input1), np.asarray(input2), input1.sd, input2.sd)
            result_units = units_func(input1.units, input2.units)
            return qnArrayTwo(result_val, units = Units('1'), sd = result_sd)

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

    def convert_units_to(self, new_units, force=None):
        to_units = Units(new_units)
        fac = self.units.conversion(to_units, force)
        new_vals = np.asarray(self)*fac
        new_sd = self.sd*fac
        return qnArrayTwo(new_vals, units = new_units, sd = new_sd)

    def append(self, input_quantity):
        if self.units != input_quantity.units:
            raise qnArrayTwoError('Same units expected')
        return qnArrayTwo(np.append(np.asarray(self),input_quantity.value), units = self.units, sd = np.append(self.sd,input_quantity.sd))

def plot_qn_arrays(qn_arr_1, qn_arr_2):
    plt.errorbar(np.asarray(qn_arr_1),np.asarray(qn_arr_2),xerr = qn_arr_1.sd, yerr = qn_arr_2.sd)
    plt.show()
