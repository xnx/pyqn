import unittest
from ..dimensions import Dimensions

class DimensionsCheck(unittest.TestCase):
    def test_dimensions_init(self):
        d1 = Dimensions(dims = [1,0,0,0,0,0,1])
        d2 = Dimensions(dims = [0,0,0,0,0,0,0])
        d3 = Dimensions(dims = [1,1,1,1,1,1,1])
    
if __name__ == '__main__':
    unittest.main()
