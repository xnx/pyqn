from pyqn.units import Units

u1 = Units('m.s-1')
u2 = Units('cm.hr-1/m')
print 'u1: ',u1
print 'u2: ',u2
print 'u1*u2 =',u1*u2
print 'u1: ',u1
print 'u2: ',u2
print 'u1/u2 =',u1/u2
print 'u2/u1 =',u2/u1

u3 = Units('J.A-1')
u4 = Units('J.A-1')
print 'u3/u4 =', u3/u4
