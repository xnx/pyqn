# pyqn
A Python package for handling physical units and quantities.

A very quick overview:

    In [1]: from pyqn.units import Units

    In [2]: u1 = Units('km')

    In [3]: u2 = Units('hr')

    In [4]: u3 = u1/u2

    In [5]: print(u3)
    km.hr-1

    In [6]: u4 = Units('m/s')

    In [7]: u3.conversion(u4)        # OK: can convert from km/hr to m/s
    Out[7]: 0.2777777777777778

    In [8]: u3.conversion(u2)        #  Oops: can't convert from km/hr to m!
    ...
    UnitsError: Failure in units conversion: units km.hr-1[L.T-1] and hr[T] have
    different dimensions

For more information and examples, see http://christianhill.co.uk/projects/pyqn
