from .units import Units, UnitsError
import numpy as np

def sd_add_sub(result, vals1, vals2, sd1, sd2):
    return np.sqrt(sd1**2+sd2**2)
def sd_mul_div(result, vals1, vals2, sd1, sd2):
    return result*np.sqrt((sd1/vals1)**2+(sd2/vals2)**2)

def sd_logaddexp(result, vals1, vals2, sd1, sd2):
    return np.sqrt(np.exp(vals1)**2*sd1**2+np.exp(vals2)**2*sd2**2)/(np.exp(vals1)+np.exp(vals2))
def sd_logaddexp2(result, vals1, vals2, sd1, sd2):
    return np.sqrt(np.exp(vals1)**2*sd1**2+np.exp(vals2)**2*sd2**2)/((np.log(2))*(np.exp(vals1)+np.exp(vals2)))
def sd_power(result, vals1, vals2, sd1, sd2):
    return np.sqrt(sd1**2*(vals2*vals1**(vals2-1))**2 + sd2**2*(result*np.log(vals1))**2)

def sd_exp(result, vals, sd):
    return result * sd
def sd_sin(result, vals, sd):
    return np.cos(vals) * sd
def sd_cos(result, vals, sd):
    return np.sin(vals) * sd
def sd_tan(result, vals, sd):
    return np.cos(np.asarray(vals))**(-2) * np.asarray(sd)
def sd_arcsin_arccos(result, vals, sd):
    return np.asarray(sd)/np.sqrt(1-np.asarray(vals)**2)
def sd_arctan(result, vals, sd):
    return np.asarray(sd)/(1.0+np.asarray(vals)**2)
def sd_sinh(result, vals, sd):
    return sd*np.cosh(vals)
def sd_cosh(result, vals, sd):
    return sd*np.sinh(vals)
def sd_tanh(result, vals, sd):
    return np.asarray(sd)*np.cosh(np.asarray(vals))**(-2)
def sd_arcsinh(result, vals, sd):
    return np.asarray(sd)/np.sqrt(1+np.asarray(vals)**2)
def sd_arccosh(result, vals, sd):
    return sd/(np.sqrt(vals-1)*np.sqrt(vals+1))
def sd_arctanh(result, vals, sd):
    return np.asarray(sd)/(1.0-np.asarray(vals)**2)
def sd_nochange(result, vals, sd):
    return sd

def units_check_unitless(input_arr):
    if input_arr.units.has_units() is True:
        raise UnitsError('qnArray must be unitless')
    else:
        return input_arr

def units_check_unitless_deg_rad(input_arr):
    if input_arr.units == Units('deg'):
        return input_arr.convert_units_to('rad')
    elif (input_add.units == Units('rad')) or (input_arr.units.has_units() is False):
        return input_arr
    else:
        raise UnitsError('qnArray must have units: deg, rad, unitless')

def units_check_any(input_arr):
    return input_arr

def units_add_sub(u1, u2):
    if u1.has_units() is True:
        return u1
    else:
        return u2
def units_mul(u1,u2):
    return u1*u2
def units_div(u1,u2):
    return u1/u2
def units_unitless(u):
    return Units('1')
def units_self(u):
    return u
def units_unitless2(u1,u2):
    return Units('1')

ufunc_dict_alg = {  np.add: ('__add__', sd_add_sub, units_add_sub, '__radd__'),
                    np.subtract: ('__sub__', sd_add_sub, units_add_sub, '__rsub__'),
                    np.multiply: ('__mul__', sd_mul_div, units_mul, '__rmul__'),
                    np.divide: ('__truediv__', sd_mul_div, units_div, '__rtruediv__')}

ufunc_dict_one_input = {np.exp: (sd_exp, units_check_unitless, units_unitless),
                        np.sin: (sd_sin, units_check_unitless_deg_rad, units_unitless),
                        np.cos: (sd_cos, units_check_unitless_deg_rad, units_unitless),
                        np.tan: (sd_tan, units_check_unitless_deg_rad, units_unitless),
                        #np.arcsin: (sd_arcsin_arccos, units_check_unitless, units_unitless),
                        #np.arccos: (sd_arcsin_arccos, units_check_unitless, units_unitless),
                        #np.arctan: (sd_arctan, units_check_unitless, units_unitless),
                        #np.sinh: (sd_sinh, units_check_unitless, units_unitless),
                        #np.cosh: (sd_cosh, units_check_unitless, units_unitless),
                        #np.tanh: (sd_tanh, units_check_unitless, units_unitless),
                        #np.arcsinh: (sd_arcsinh, units_check_unitless, units_unitless),
                        #np.arccosh: (sd_arccosh, units_check_unitless, units_unitless),
                        #np.arctanh: (sd_arctanh, units_check_unitless, units_unitless),
                        #np.negative: (sd_nochange, units_check_any, units_self)
                        }

ufunc_dict_two_inputs = {np.logaddexp: (sd_logaddexp, units_check_unitless, units_unitless2),
                         np.logaddexp2: (sd_logaddexp2, units_check_unitless, units_unitless2),
                         np.power: (sd_power, units_check_unitless, units_unitless2),
                         np.true_divide: (sd_mul_div, units_check_any, units_div),
                         np.floor_divide: (sd_mul_div, units_check_any, units_div)}

