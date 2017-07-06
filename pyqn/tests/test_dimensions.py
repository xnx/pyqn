import unittest
from ..dimensions import Dimensions, DimensionsError

class DimensionsCheck(unittest.TestCase):
    def test_dimensions_init(self):
        arr1 = [1,0,0,0,0,0,1]
        arr2 = [0,2,10,0,5,0,0]
        arr3 = [1,1,-1,1,-1,1,-1]
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
    
    def test_dimensions_mul(self):
        arr1 = [1,0,0,0,0,0,0]
        arr2 = [0,1,1,0,1,0,0]
        arr3 = [-1,10,5,2,0,-4,0]
        d1 = Dimensions(dims = arr1)
        d2 = Dimensions(dims = arr2)
        d3 = Dimensions(dims = arr3)

        d12 = d1 * d2
        d13 = d1 * d3
        d23 = d2 * d3
        for i in range(7):
            self.assertEqual(d12.dims[i], arr1[i]+arr2[i])
            self.assertEqual(d13.dims[i], arr1[i]+arr3[i])
            self.assertEqual(d23.dims[i], arr2[i]+arr3[i])

    def test_dimensions_div(self):
        arr1 = [1,0,0,0,0,0,0]
        arr2 = [0,1,1,0,1,0,0]
        arr3 = [-1,10,5,2,0,-4,0]
        d1 = Dimensions(dims = arr1)
        d2 = Dimensions(dims = arr2)
        d3 = Dimensions(dims = arr3)

        d12 = d1 / d2
        d21 = d2 / d1
        d13 = d1 / d3
        d31 = d3 / d1
        d23 = d2 / d3
        d32 = d3 / d2
        for i in range(7):
            self.assertEqual(d12.dims[i], arr1[i]-arr2[i])
            self.assertEqual(d21.dims[i], arr2[i]-arr1[i])
            self.assertEqual(d13.dims[i], arr1[i]-arr3[i])
            self.assertEqual(d31.dims[i], arr3[i]-arr1[i])
            self.assertEqual(d23.dims[i], arr2[i]-arr3[i])
            self.assertEqual(d32.dims[i], arr3[i]-arr2[i])
        

if __name__ == '__main__':
    unittest.main()
