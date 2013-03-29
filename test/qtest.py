from quantity import Quantity

q1 = Quantity(name='E1', value=4.2e-19, units='J', sd=1.e-20)
q2 = Quantity(name='E2', value=3.7e-19, units='J', sd=1.5e-20)
print q1, q1.sd
print q2, q2.sd
q3 = q1 + q2
print q3, q3.sd
q4 = q1*q2
print q4, q4.sd

q1.convert_units_to('eV')
print q1, q1.sd

print q3-q2
print q4-q1
