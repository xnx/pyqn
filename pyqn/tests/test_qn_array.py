import unittest
from ..qn_array import QnArray, QnArrayError
from ..units import Units
from ..quantity import Quantity
from ..ufunc_dictionaries import *
import numpy as np

class QnArrayTest(unittest.TestCase):
    def test_qn_array_init(self):
        vals1 = [1,2,3,4]
        sd1 = [0.1,0.2,0.3,0.4]
        qnarr1 = QnArray(vals1,units='m',sd=sd1)
        self.assertEqual(qnarr1.units, Units('m'))
        for i in range(len(vals1)):
            self.assertEqual(qnarr1[i], vals1[i])
            self.assertEqual(qnarr1.sd[i], sd1[i])
            
        vals2 = [-10,-20,-30,0,5]
        qnarr2 = QnArray(vals2, units = 'J')
        self.assertEqual(qnarr2.units, Units('J'))
        for i in range(len(vals2)):
            self.assertEqual(qnarr2[i], vals2[i])
            self.assertEqual(qnarr2.sd[i], 0)

    def test_qn_array_add(self):
        vals1 = [1,2,3,4]
        sd1 = [0.1,0.2,0.3,0.4]
        qnarr1 = QnArray(vals1,units='m',sd=sd1)
        
        vals2 = [2,3,4,1]
        sd2 = [0.2,0.3,0.4,0.1]
        qnarr2 = QnArray(vals2, units='m',sd=sd2)
        
        qnarr3 = qnarr1 + qnarr2
        self.assertEqual(qnarr3.units, qnarr1.units)
        for i in range(len(vals2)):
            self.assertEqual(qnarr3[i], vals1[i]+vals2[i])
            self.assertAlmostEqual(qnarr3.sd[i], (sd1[i]**2+sd2[i]**2)**0.5)
            
        vals3 = [1,1,1]
        qnarr3 = QnArray(vals3, units='m')
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
            
    def test_qn_array_sub(self):
        vals1 = [1,2,3,4]
        qnarr1 = QnArray(vals1,units='m')
        
        vals2 = [2,3,4,1]
        qnarr2 = QnArray(vals2, units='m')
        
        qnarr3 = qnarr1 - qnarr2
        for i in range(len(vals2)):
            self.assertEqual(qnarr3[i], vals1[i]-vals2[i])
            
        vals3 = [1,1,1]
        qnarr3 = QnArray(vals3, units='m')
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

    def test_qn_array_mul(self):
        vals1 = [1,2,3,4]
        qnarr1 = QnArray(vals1,units='m')
        
        vals2 = [2,3,4,1]
        qnarr2 = QnArray(vals2, units='s')
        
        qnarr3 = qnarr1 * qnarr2
        self.assertEqual(qnarr3.units, Units('m.s'))
        for i in range(len(vals2)):
            self.assertEqual(qnarr3[i], vals1[i]*vals2[i])
            
        vals3 = [1,1,1]
        qnarr3 = QnArray(vals3, units='m')
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

    def test_qn_array_mul(self):
        vals1 = [1,2,3,4]
        qnarr1 = QnArray(vals1,units='m')
        
        vals2 = [2,3,4,1]
        qnarr2 = QnArray(vals2, units='s')
        
        qnarr3 = qnarr1 / qnarr2
        self.assertEqual(qnarr3.units, Units('m.s-1'))
        for i in range(len(vals2)):
            self.assertEqual(qnarr3[i], vals1[i]/vals2[i])
            
        vals3 = [1,1,1]
        qnarr3 = QnArray(vals3, units='m')
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
            
    def test_qn_array_conversion(self):
        a1 = [1,2,3]
        sd1 = [0.1,0.2,0.3]
        q1 = QnArray(a1, units = 'm', sd = sd1)
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
        q1 = QnArray(a1, units = 'm', sd = sd1)
        quant1 = Quantity(value=10,units='m')
        quant2 = Quantity(value=5,units='m',sd=0.5)
        quant3 = Quantity(value=2,units='J',sd=0.1)
        
        result1 = q1.append(quant1)
        result2 = q1.append(quant2)
        with self.assertRaises(QnArrayError) as e:
            result3 = q1.append(quant3)
        self.assertEqual(len(result1),4)
        self.assertEqual(result1[-1],10)
        self.assertEqual(len(result2),4)
        self.assertEqual(result2[-1],5)
        
    def test_qn_array_ufunc(self):
        a1 = [1,2,3]
        sd1 = [0.1,0.2,0.3]
        a2 = [0.1,0.2,0.3]
        sd2 = [0.01,0.02,0.03]
        q1 = QnArray(a1,units = 'm', sd = sd1)
        q2 = QnArray(a2,units = 'm', sd = sd2)
        q3 = QnArray(a2,units = 's', sd = sd2)
        
        for ufunc in ufunc_dict_alg:
            if (ufunc is np.multiply) or (ufunc is np.divide):
                in1 = q1
                in2 = q3
            elif (ufunc is np.add) or (ufunc is np.subtract):
                in1 = q1
                in2 = q2
            result = ufunc(in1, in2)
            
            sd_func = ufunc_dict_alg[ufunc][1]
            units_func = ufunc_dict_alg[ufunc][2]
            
            self.assertTrue(result.units, units_func(in1.units, in2.units))
            for i in range(3):
                self.assertEqual(result[i], ufunc(np.asarray(in1), np.asarray(in2))[i])
                self.assertEqual(result.sd[i], sd_func(np.asarray(result), np.asarray(in1), np.asarray(in2), in1.sd, in2.sd)[i])
                
            print('{} tested!'.format(ufunc))
        
        a1 = [0.1,0.2,0.3]
        sd1 = [0.01,0.02,0.03]
        a2 = [0.4,0.5,0.6]
        sd2 = [0.04,0.05,0.06]
        q3 = QnArray(a1, units = '1', sd = sd1)
        q4 = QnArray(a1, units = 'rad', sd = sd1)
        q5 = QnArray(a1, units = 'deg', sd = sd1)
        q6 = QnArray(a2, units = '1', sd = sd2)
        
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
            elif unit_test is units_check_any:
                in1 = [unit_test(q1), unit_test(q2), unit_test(q3), unit_test(q4), unit_test(q5)]
                r = [ufunc(i) for i in in1]
            sd_func = ufunc_dict_one_input[ufunc][0] #(result, vals, sd)
            units_func = ufunc_dict_one_input[ufunc][2] #(units)
            
            for result, input1 in zip(r,in1):
                self.assertEqual(result.units, units_func(input1.units))
                for i in range(3):
                    self.assertAlmostEqual(result[i], ufunc(np.asarray(input1))[i], places = 2)
                    self.assertAlmostEqual(result.sd[i], sd_func(result, input1, input1.sd)[i],places = 1)
            
            print('{} tested!'.format(ufunc))

        for ufunc in ufunc_dict_two_inputs:
            unit_test = ufunc_dict_two_inputs[ufunc][1]
            if unit_test is units_check_unitless:
                with self.assertRaises(Exception) as e:
                    ufunc(q1,q3)
                with self.assertRaises(Exception) as e:
                    ufunc(q2,q6)
                with self.assertRaises(Exception) as e:
                    ufunc(q4,q3)
                with self.assertRaises(Exception) as e:
                    ufunc(q5,q3)
                in1 = [unit_test(q3)]
                in2 = [unit_test(q6)]
                r = [ufunc(i1,i2) for i1,i2 in zip(in1,in2)]
            elif unit_test is units_check_any:
                in1 = [unit_test(q1), unit_test(q2), unit_test(q3), unit_test(q4), unit_test(q5), unit_test(q6)]
                in2 = [unit_test(q6), unit_test(q5), unit_test(q4), unit_test(q3), unit_test(q2), unit_test(q1)]
                r = [ufunc(i1,i2) for i1,i2 in zip(in1,in2)]
            sd_func = ufunc_dict_two_inputs[ufunc][0]
            units_func = ufunc_dict_two_inputs[ufunc][2]
                
            for result, input1, input2 in zip(r, in1, in2):
                self.assertEqual(result.units, units_func(input1.units, input2.units))
                result_got = result
                result_wanted = ufunc(np.asarray(input1),np.asarray(input2))
                sd_got = result.sd
                sd_wanted = sd_func(np.asarray(result), 
                                    np.asarray(input1), np.asarray(input2),
                                    input1.sd, input2.sd)
                for i in range(3):
                    self.assertAlmostEqual(result_got[i], result_wanted[i])
                    self.assertAlmostEqual(sd_got[i], sd_wanted[i])
            
            print('{} tested!'.format(ufunc))

if __name__ == '__main__':
    unittest.main()
