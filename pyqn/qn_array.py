#!/usr/bin/env python
# -*- coding: utf-8 -*-

# qn_array.py
# A class representing an array of physical quantities, with units and 
# uncertainty.
#
# Copyright (C) 2012-2017 Christian Hill
# Department of Physics and Astronomy, University College London
# christian.hill@ucl.ac.uk
#
# This file is part of PyQn
#
# PyQn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyQn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyQn.  If not, see <http://www.gnu.org/licenses/>

import numpy as np
from .units import Units
from .quantity import Quantity
from .ufunc_dictionaries import *
import matplotlib.pyplot as plt
import csv

class QnArrayError(Exception):
    """
    An Exception class for errors that might occur whilst manipulating
    QnArray objects.
    
    """
    
    def __init__(self, error_str):
        self.error_str = error_str
    def __str__(self):
        return self.error_str

class QnArray(np.ndarray):
    """
    A Python class representing an array of values, their units and 
    errors. This class extends the numpy ndarray class such that 
    QnArray objects can be used with numpy ufuncs and the errors are 
    propagated using the functions defined in the ufunc_dictionaries.py
    file.
    
    """
    
    def __new__(cls, input_array, info=None, units='1', sd=None):
        """ 
        Initialises a qnArray as a child class derived from the numpy 
        ndarray. The units are set to default to be unitless, but can
        be set to be a string (which will be converted to a Units 
        object) or set straight away to be a Units object, the standard
        deviation is set to default to an array of zeros.
        
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
        """
        Function which is called each time that a function is operated 
        on the QnArray object.
        
        """
        
        if obj is None: return
        self.info = getattr(obj, 'info', None)
        self.sd = getattr(obj, 'sd', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Function which is called when numpy ufuncs are called on the 
        QnArray object. It uses predefined equations to calculate the
        new errors of the returned QnArray objects.
        
        """
        
        if ufunc in ufunc_dict_alg:
            alg_func = ufunc_dict_alg[ufunc][0]
            sd_func = ufunc_dict_alg[ufunc][1]
            units_func = ufunc_dict_alg[ufunc][2]
            alg_func_reverse = ufunc_dict_alg[ufunc][3]

            if all([hasattr(x, 'units') for x in inputs]):
                result_units = units_func(inputs[0].units, 
                                          inputs[1].units)

            # if both are qn arrays
            if (type(inputs[1]) is QnArray) and \
                (type(inputs[0]) is QnArray):
                result_val = getattr(np.asarray(inputs[0]), 
                                     alg_func)(np.asarray(inputs[1]))
                result_sd = sd_func(result_val, np.asarray(inputs[0]),
                                                np.asarray(inputs[1]),
                                                inputs[0].sd,
                                                inputs[1].sd)

            # if one input is a quantity
            elif type(inputs[1]) is Quantity:
                result_val = getattr(np.asarray(inputs[0]), 
                                     alg_func)(inputs[1].value)
                result_sd = sd_func(result_val, np.asarray(inputs[0]),
                                                inputs[1].value,
                                                inputs[0].sd,
                                                inputs[1].sd)

            elif type(inputs[0]) is Quantity:
                result_val = getattr(np.asarray(inputs[1]), 
                                     alg_func_reverse)(inputs[0].value)
                result_sd = sd_func(result_val, inputs[0].value,
                                                np.asarray(inputs[1]),
                                                inputs[0].sd,
                                                inputs[1].sd)

            # for all other object types
            elif type(inputs[0]) is QnArray:
                result_val = getattr(np.asarray(inputs[0]), 
                                     alg_func)(inputs[1])
                result_sd = sd_func(result_val, np.asarray(inputs[0]),
                                                inputs[1],
                                                inputs[0].sd, 0)
                result_units = units_func(inputs[0].units, Units('1'))

            else:
                result_val = getattr(np.asarray(inputs[1]), 
                                     alg_func_reverse)(inputs[0])
                result_sd = sd_func(result_val, inputs[0],
                                                np.asarray(inputs[1]),
                                                0, inputs[1].sd)
                result_units = units_func(Units('1'), inputs[1].units)

            return QnArray(result_val, units = result_units, 
                                       sd = result_sd)
            
        elif ufunc in ufunc_dict_one_input:
            #checks if units of input are valid
            unit_test_func = ufunc_dict_one_input[ufunc][1]
            inputs_checked = unit_test_func(inputs[0])
            
            #extracts functions for finding sd and units
            sd_func = ufunc_dict_one_input[ufunc][0]
            units_func = ufunc_dict_one_input[ufunc][2]
            
            #calculates results
            result_val = ufunc(np.asarray(inputs_checked))
            result_sd = sd_func(result_val, 
                                np.asarray(inputs_checked), 
                                np.asarray(inputs_checked.sd))
            result_units = units_func(inputs[0].units)
            return QnArray(result_val, units = result_units, 
                                       sd = result_sd)

        elif ufunc in ufunc_dict_two_inputs:
            unit_test_func = ufunc_dict_two_inputs[ufunc][1]
            input1 = unit_test_func(inputs[0])
            input2 = unit_test_func(inputs[1])
            sd_func = ufunc_dict_two_inputs[ufunc][0]
            units_func = ufunc_dict_two_inputs[ufunc][2]
            result_val = ufunc(np.asarray(input1),np.asarray(input2))
            result_sd = sd_func(result_val, np.asarray(input1), 
                                            np.asarray(input2), 
                                            input1.sd, input2.sd)
            result_units = units_func(input1.units, input2.units)
            return QnArray(result_val, units = result_units, 
                                       sd = result_sd)

    def __eq__(self, other):
        """
        Checks if two QnArray objects are the same, given that their 
        values and units are the same.
        
        """
        
        if all(super(QnArray, self).__eq__(super(QnArray, other))) and \
            (self.units == other.units):
            return True
        else:
            return False

    @property
    def html_str(self):
        """
        Creates a representation of the array in the HTML format.
        
        """
        
        html_chunks = []
        for i in range(len(self)):
            html_chunks.append('{} {}'.format(self[i],self.units))
        return ', '.join(html_chunks)

    def convert_units_to(self, new_units, force=None):
        """
        Converts the units of the array given a new unit. It uses the 
        conversion factor generator of the Units class.
        
        """
        
        to_units = Units(new_units)
        fac = self.units.conversion(to_units, force)
        new_vals = np.asarray(self)*fac
        new_sd = self.sd*fac
        return QnArray(new_vals, units = new_units, sd = new_sd)

    def append(self, input_quantity):
        """
        Appends a Quantity object to the QnArray object.
        
        """
        
        if self.units != input_quantity.units:
            raise QnArrayError('Same units expected')
        return QnArray(np.append(np.asarray(self),
                       input_quantity.value), 
                       units = self.units, 
                       sd = np.append(self.sd,input_quantity.sd))

def plot_qn_arrays(qn_arr_1, qn_arr_2):
    """
    Plots two QnArray objects using matplotlib with their errors.
    
    """
    
    plt.errorbar(np.asarray(qn_arr_1),np.asarray(qn_arr_2),
                    xerr = qn_arr_1.sd, yerr = qn_arr_2.sd)
    plt.show()
    
def load_data(filename, file_type, errors=False):
    """
    Function which generates two QnArray objects given a csv file.
    
    """
    
    vals1 = []
    vals2 = []
    sd1 = []
    sd2 = []
    if file_type is 'csv':
        with open(filename,'rb') as f:
            reader = csv.reader(f)
            if errors is False:
                for row in reader:
                    vals1.append(row[0])
                    vals2.append(row[1])
                    sd1.append(0)
                    sd2.append(0)
            else:
                for row in reader:
                    vals1.append(row[0])
                    sd1.append(row[1])
                    vals2.append(row[2])
                    sd2.append(row[3])
    return QnArray(vals1, units = '1', sd = sd1), \
            QnArray(vals2, units = '1', sd = sd2)
