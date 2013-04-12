import sys
from pyqn.quantity import Quantity


A = Quantity.parse('A = 12.3(4) m2')
B = Quantity.parse('B = 2.10(5) m')
C = B*A; C.name = 'C = B*A'
print A,B,C
print A.units.get_dims(), B.units.get_dims(), C.units.get_dims()

D = B/A; D.name = 'D = B/A'
print A.as_str(), B.as_str(), D.as_str()
print A.units.get_dims(), B.units.get_dims(), D.units.get_dims()

DeltaH = Quantity.parse('DeltaH = 47.15(4) kJ.mol-1')
T = Quantity(name='T', units='K', value=298., sd=0.5)
DeltaS = DeltaH / T
DeltaS.convert_units_to('J.K-1.mol-1')
print DeltaS.as_str()
print DeltaS.units.get_dims()
DeltaS2 = Quantity.parse('DeltaS2 = -75.24(25) J.K-1.mol-1')
DeltaS3 = DeltaS + DeltaS2
print DeltaS3
print DeltaS3.as_str()

g = Quantity.parse('g = 9.818(7) m.s-2')
print g
print g.as_str()
print g.as_str(b_sd=False)
print g.as_str(b_name=False, b_sd=False)

sys.exit(0)

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
#print q4-q1

print '----'

ss = ['E1 = 1.342(7)D-03 J', '2309.32(400)', '-1.23443(9)e+05 meV/T',
      'r = +1.032(23)e-6', 'charlotte=-232.434(90) mmHg-1.A',
      '100466(7) nJ-1', 'emma =34.5e-2', '4.002602(2)']
for s in ss:
    p = Quantity.parse(s)
    print '"%s"' % s, p, p.sd
    print 'as_str():', p.as_str()
