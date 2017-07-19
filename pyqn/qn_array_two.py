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
        if ufunc in ufunc_dict_alg:
            alg_func = ufunc_dict_alg[ufunc][0]
            sd_func = ufunc_dict_alg[ufunc][1]
            units_func = ufunc_dict_alg[ufunc][2]
            alg_func_reverse = ufunc_dict_alg[ufunc][3]
            
            # check for units matching
            #if hasattr(inputs[0],'units') and hasattr(inputs[1],'units'):
            #    if inputs[0].units != inputs[1].units:
            #        raise qnArrayTwoError('Units must match')
            
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
        elif ufunc in ufunc_dict_other:
            if inputs[0].units.has_units() is True:
                raise qnArrayTwoError('qnArray must be unitless')
            sd_func = ufunc_dict_other[ufunc]
            result_val = np.exp(super(qnArrayTwo, inputs[0]))
            result_sd = sd_func(super(qnArrayTwo, result_val), super(qnArrayTwo, inputs[0]), inputs[0].sd)
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
                
def sd_add_sub(result, vals1, vals2, sd1, sd2):
    return np.sqrt(sd1**2+sd2**2)
def sd_mul_div(result, vals1, vals2, sd1, sd2):
    return result*np.sqrt((sd1/vals1)**2+(sd2/vals2)**2)
def sd_exp(result, vals, sd):
    return result * sd
    
def units_add_sub(u1, u2):
	if u1.has_units() is True:
		return u1
	else:
		return u2
def units_mul(u1,u2):
    return u1*u2
def units_div(u1,u2):
    return u1/u2
    
ufunc_dict_alg = {  np.add: ('__add__', sd_add_sub, units_add_sub, '__radd__'),
                    np.subtract: ('__sub__', sd_add_sub, units_add_sub, '__rsub__'),
                    np.multiply: ('__mul__', sd_mul_div, units_mul, '__rmul__'),
                    np.divide: ('__truediv__', sd_mul_div, units_div, '__rtruediv__')}
                
ufunc_dict_other = { np.exp: sd_exp}
