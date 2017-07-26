import unittest
from ..qn_array_two import qnArrayTwo, qnArrayTwoError
from ..units import Units
from ..quantity import Quantity
from ..ufunc_dictionaries import *
import numpy as np

class qnArrayTwoTest(unittest.TestCase):
    def test_qn_array_two_init(self):
        vals1 = [1,2,3,4]
        sd1 = [0.1,0.2,0.3,0.4]
        qnarr1 = qnArrayTwo(vals1,units='m',sd=sd1)
        self.assertEqual(qnarr1.units, Units('m'))
        for i in range(len(vals1)):
            self.assertEqual(qnarr1[i], vals1[i])
            self.assertEqual(qnarr1.sd[i], sd1[i])
            
        vals2 = [-10,-20,-30,0,5]
        qnarr2 = qnArrayTwo(vals2, units = 'J')
        self.assertEqual(qnarr2.units, Units('J'))
        for i in range(len(vals2)):
            self.assertEqual(qnarr2[i], vals2[i])
            self.assertEqual(qnarr2.sd[i], 0)

    def test_qn_array_two_add(self):
        vals1 = [1,2,3,4]
        sd1 = [0.1,0.2,0.3,0.4]
        qnarr1 = qnArrayTwo(vals1,units='m',sd=sd1)
        
        vals2 = [2,3,4,1]
        sd2 = [0.2,0.3,0.4,0.1]
        qnarr2 = qnArrayTwo(vals2, units='m',sd=sd2)
        
        qnarr3 = qnarr1 + qnarr2
        self.assertEqual(qnarr3.units, qnarr1.units)
        for i in range(len(vals2)):
            self.assertEqual(qnarr3[i], vals1[i]+vals2[i])
            self.assertAlmostEqual(qnarr3.sd[i], (sd1[i]**2+sd2[i]**2)**0.5)
            
        vals3 = [1,1,1]
        qnarr3 = qnArrayTwo(vals3, units='m')
        with self.assertRaises(ValueError) as e:
            qnarr = qnarr1 + qnarr3

        q1 = Quantity(value = 10, units = 'm')
        qnarr4 = qnarr1 + q1
        qnarr5 = q1 + qnarr1
        qnarr6 = qnarr2 + 15
        qnarr7 = 15 + qnarr2
        self.assertEqual(qnarr4.units, Units('m'))
        self.assertEqual(qnarr5.units, Units('m'))
        self.assertEqual(qnarr6.units, Units('m'))
        self.assertEqual(qnarr7.units, Units('m'))
        for i in range(len(qnarr4)):
            self.assertEqual(qnarr4[i], qnarr1[i]+10)
            self.assertEqual(qnarr5[i], 10+qnarr1[i])
            self.assertEqual(qnarr6[i], qnarr2[i]+15)
            self.assertEqual(qnarr7[i], 15+qnarr2[i])
            
    def test_qn_array_two_sub(self):
        vals1 = [1,2,3,4]
        qnarr1 = qnArrayTwo(vals1,units='m')
        
        vals2 = [2,3,4,1]
        qnarr2 = qnArrayTwo(vals2, units='m')
        
        qnarr3 = qnarr1 - qnarr2
        for i in range(len(vals2)):
            self.assertEqual(qnarr3[i], vals1[i]-vals2[i])
            
        vals3 = [1,1,1]
        qnarr3 = qnArrayTwo(vals3, units='m')
        with self.assertRaises(ValueError) as e:
            qnarr = qnarr1 - qnarr3

        q1 = Quantity(value = 10, units = 'm')
        qnarr4 = qnarr1 - q1
        qnarr5 = q1 - qnarr1
        qnarr6 = qnarr2 - 15
        qnarr7 = 15 - qnarr2
        self.assertEqual(qnarr4.units, Units('m'))
        self.assertEqual(qnarr5.units, Units('m'))
        self.assertEqual(qnarr6.units, Units('m'))
        self.assertEqual(qnarr7.units, Units('m'))
        for i in range(len(qnarr4)):
            self.assertEqual(qnarr4[i], qnarr1[i]-10)
            self.assertEqual(qnarr5[i], 10-qnarr1[i])
            self.assertEqual(qnarr6[i], qnarr2[i]-15)
            self.assertEqual(qnarr7[i], 15-qnarr2[i])

    def test_qn_array_two_mul(self):
        vals1 = [1,2,3,4]
        qnarr1 = qnArrayTwo(vals1,units='m')
        
        vals2 = [2,3,4,1]
        qnarr2 = qnArrayTwo(vals2, units='s')
        
        qnarr3 = qnarr1 * qnarr2
        self.assertEqual(qnarr3.units, Units('m.s'))
        for i in range(len(vals2)):
            self.assertEqual(qnarr3[i], vals1[i]*vals2[i])
            
        vals3 = [1,1,1]
        qnarr3 = qnArrayTwo(vals3, units='m')
        with self.assertRaises(ValueError) as e:
            qnarr = qnarr1 * qnarr3

        q1 = Quantity(value = 10, units = 'J')
        qnarr4 = qnarr1 * q1
        qnarr5 = q1 * qnarr1
        qnarr6 = qnarr2 * 15
        qnarr7 = 15 * qnarr2
        self.assertEqual(qnarr4.units, Units('m.J'))
        self.assertEqual(qnarr5.units, Units('m.J'))
        self.assertEqual(qnarr6.units, Units('s'))
        self.assertEqual(qnarr7.units, Units('s'))
        for i in range(len(qnarr4)):
            self.assertEqual(qnarr4[i], qnarr1[i]*10)
            self.assertEqual(qnarr5[i], 10*qnarr1[i])
            self.assertEqual(qnarr6[i], qnarr2[i]*15)
            self.assertEqual(qnarr7[i], 15*qnarr2[i])

    def test_qn_array_two_mul(self):
        vals1 = [1,2,3,4]
        qnarr1 = qnArrayTwo(vals1,units='m')
        
        vals2 = [2,3,4,1]
        qnarr2 = qnArrayTwo(vals2, units='s')
        
        qnarr3 = qnarr1 / qnarr2
        self.assertEqual(qnarr3.units, Units('m.s-1'))
        for i in range(len(vals2)):
            self.assertEqual(qnarr3[i], vals1[i]/vals2[i])
            
        vals3 = [1,1,1]
        qnarr3 = qnArrayTwo(vals3, units='m')
        with self.assertRaises(ValueError) as e:
            qnarr = qnarr1 / qnarr3

        q1 = Quantity(value = 10, units = 'J')
        qnarr4 = qnarr1 / q1
        qnarr5 = q1 / qnarr1
        qnarr6 = qnarr2 / 15
        qnarr7 = 15 / qnarr2
        self.assertEqual(qnarr4.units, Units('m.J-1'))
        self.assertEqual(qnarr5.units, Units('m-1.J'))
        self.assertEqual(qnarr6.units, Units('s'))
        self.assertEqual(qnarr7.units, Units('s-1'))
        for i in range(len(qnarr4)):
            self.assertEqual(qnarr4[i], qnarr1[i]/10)
            self.assertEqual(qnarr5[i], 10/qnarr1[i])
            self.assertEqual(qnarr6[i], qnarr2[i]/15)
            self.assertEqual(qnarr7[i], 15/qnarr2[i])
            
    def test_qn_array_two_conversion(self):
        a1 = [1,2,3]
        sd1 = [0.1,0.2,0.3]
        q1 = qnArrayTwo(a1, units = 'm', sd = sd1)
        q2 = q1.convert_units_to('inch')
        r_wanted = [a*39.37007874 for a in a1]
        sd_wanted = [s*39.37007874 for s in sd1]
        
        self.assertEqual(q2.units, Units('inch'))
        for i in range(3):
            self.assertAlmostEqual(q2[i], r_wanted[i])
            self.assertAlmostEqual(q2.sd[i], sd_wanted[i])
            
    def test_qn_array_append(self):
        a1 = [1,2,3]
        sd1 = [0.1,0.2,0.3]
        q1 = qnArrayTwo(a1, units = 'm', sd = sd1)
        quant1 = Quantity(value=10,units='m')
        quant2 = Quantity(value=5,units='m',sd=0.5)
        quant3 = Quantity(value=2,units='J',sd=0.1)
        
        result1 = q1.append(quant1)
        result2 = q1.append(quant2)
        with self.assertRaises(qnArrayTwoError) as e:
            result3 = q1.append(quant3)
        self.assertEqual(len(result1),4)
        self.assertEqual(result1[-1],10)
        self.assertEqual(len(result2),4)
        self.assertEqual(result2[-1],5)
        
    def test_qn_array_two_ufunc(self):
        a1 = [1,2,3]
        sd1 = [0.1,0.2,0.3]
        a2 = [0.1,0.2,0.3]
        sd2 = [0.01,0.02,0.03]
        q1 = qnArrayTwo(a1,units = 'm', sd = sd1)
        q2 = qnArrayTwo(a2,units = 'm', sd = sd2)
        q3 = qnArrayTwo(a2,units = 's', sd = sd2)
        
        #~ for ufunc in ufunc_dict_alg:
            #~ if (ufunc is np.multiply) or (ufunc is np.divide):
                #~ in1 = q1
                #~ in2 = q3
            #~ elif (ufunc is np.add) or (ufunc is np.subtract):
                #~ in1 = q1
                #~ in2 = q2
            #~ result = ufunc(in1, in2)
            
            #~ sd_func = ufunc_dict_alg[ufunc][1]
            #~ units_func = ufunc_dict_alg[ufunc][2]
            
            #~ self.assertTrue(result.units, units_func(in1.units, in2.units))
            #~ for i in range(3):
                #~ self.assertEqual(result[i], ufunc(np.asarray(in1), np.asarray(in2))[i])
                #~ self.assertEqual(result.sd[i], sd_func(np.asarray(result), np.asarray(in1), np.asarray(in2), in1.sd, in2.sd)[i])
                
            #~ print('{} tested!'.format(ufunc))
        
        a1 = [0.1,0.2,0.3]
        sd1 = [0.01,0.02,0.03]
        q3 = qnArrayTwo(a1, units = '1', sd = sd1)
        q4 = qnArrayTwo(a1, units = 'rad', sd = sd1)
        q5 = qnArrayTwo(a1, units = 'deg', sd = sd1)
        
        for ufunc in ufunc_dict_one_input:
            unit_test = ufunc_dict_one_input[ufunc][1]
            if unit_test is units_check_unitless:
                with self.assertRaises(Exception) as e:
                    ufunc(q1)
                with self.assertRaises(Exception) as e:
                    ufunc(q2)
                with self.assertRaises(Exception) as e:
                    ufunc(q4)
                with self.assertRaises(Exception) as e:
                    ufunc(q5)
                in1 = [unit_test(q3)]
                r = [ufunc(in1[0])]
            elif unit_test is units_check_unitless_deg_rad:
                with self.assertRaises(Exception) as e:
                    ufunc(q1)
                with self.assertRaises(Exception) as e:
                    ufunc(q2)
                in1 = [unit_test(q3), unit_test(q4), unit_test(q5)]
                r = [ufunc(in1[0]), ufunc(in1[1]), ufunc(in1[2])]
            sd_func = ufunc_dict_one_input[ufunc][0] #(result, vals, sd)
            units_func = ufunc_dict_one_input[ufunc][2] #(units)
            
            for result, input1 in zip(r,in1):
                self.assertEqual(result.units, units_func(input1.units))
                for i in range(3):
                    self.assertAlmostEqual(result[i], ufunc(np.asarray(input1))[i], places = 2)
                    self.assertAlmostEqual(result.sd[i], sd_func(result, input1, input1.sd)[i],places = 1)
            
            print('{} tested!'.format(ufunc))
        
        #~ q4 = np.exp(q3)
        #~ q5 = np.sin(q3)
        #~ q6 = np.cos(q3)
        #~ q7 = np.tan(q3)
        #~ q8 = np.arcsin(q33)
        #~ q9 = np.arccos(q33)
        #~ q10 = np.arctan(q33)
        #~ q11 = np.sinh(q33)
        #~ q12 = np.cosh(q33)
        #~ q13 = np.tanh(q33)
        #~ q14 = np.logaddexp(q3,q33)
        #~ q15 = np.power(q3,q33)
        #~ for i in range(3):
            #~ self.assertAlmostEqual(q4[i], np.exp(a1[i]))
            #~ self.assertAlmostEqual(q4.sd[i], q4[i]*q3.sd[i])
            #~ self.assertAlmostEqual(q5[i], np.sin(a1[i]))
            #~ self.assertAlmostEqual(q5.sd[i], np.cos(a1[i])*sd1[i])
            #~ self.assertAlmostEqual(q6[i], np.cos(a1[i]))
            #~ self.assertAlmostEqual(q6.sd[i], np.sin(a1[i])*sd1[i])
            #~ self.assertAlmostEqual(q7[i], np.tan(a1[i]))
            #~ #self.assertAlmostEqual(q7.sd[i], np.cos(a1[i])**(-2)*sd1[i])
            #~ self.assertAlmostEqual(q8[i], np.arcsin(a2[i]))
            #~ #self.assertAlmostEqual(q8.sd[i], sd2[i]/np.sqrt(1-sd2[i]**2))
            #~ self.assertAlmostEqual(q9[i], np.arccos(a2[i]))
            #~ #self.assertAlmostEqual(q9.sd[i], sd2[i]/np.sqrt(1-sd2[i]**2))
            #~ self.assertAlmostEqual(q10[i], np.arctan(a2[i]))
            #~ #self.assertAlmostEqual(q10.sd[i], sd2[i]*np.cosh(a2[i])**(-2))
            #~ self.assertAlmostEqual(q11[i], np.sinh(a2[i]))
            #~ self.assertAlmostEqual(q11.sd[i], sd2[i]*np.cosh(a2[i]))
            #~ self.assertAlmostEqual(q12[i], np.cosh(a2[i]))
            #~ self.assertAlmostEqual(q12.sd[i], sd2[i]*np.sinh(a2[i]))
            #~ self.assertAlmostEqual(q13[i], np.tanh(a2[i]))
            #~ self.assertAlmostEqual(q13.sd[i], sd2[i]*np.cosh(a2[i])**(-2))
            #~ self.assertAlmostEqual(q14[i], np.logaddexp(a1[i], a2[i]))
            #~ self.assertAlmostEqual(q15[i], np.power(np.asarray(q3[i]),np.asarray(q33[i])))

if __name__ == '__main__':
    unittest.main()
