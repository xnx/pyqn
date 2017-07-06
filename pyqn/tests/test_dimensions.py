import unittest
from ..dimensions import Dimensions, DimensionsError

class DimensionsCheck(unittest.TestCase):
    def test_dimensions_init(self):
        arr1 = [1,0,0,0,0,0,1]
        arr2 = [0,0,0,0,0,0,0]
        arr3 = [1,1,1,1,1,1,1]
        d1 = Dimensions(dims = arr1)
        d2 = Dimensions(dims = arr2)
        d3 = Dimensions(dims = arr3)
        d4 = Dimensions(dims = [])

        for i in range(7):
            self.assertEqual(d1.dims[i], arr1[i])
            self.assertEqual(d2.dims[i], arr2[i])
            self.assertEqual(d3.dims[i], arr3[i])
            self.assertEqual(d4.dims[i], 0)

        with self.assertRaises(DimensionsError) as e:
            d = Dimensions(dims = [1])
        with self.assertRaises(DimensionsError) as e:
            d = Dimensions(dims = [1,1,1,1,1,1,1,1,1,1,1])
    
if __name__ == '__main__':
    unittest.main()
