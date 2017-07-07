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

    def test_dimensions_truediv(self):
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
        
    def test_dimensions_pow(self):
        arr1 = [1,0,0,0,0,0,0]
        arr2 = [0,1,1,0,1,0,0]
        arr3 = [-1,10,5,2,0,-4,0]
        d1 = Dimensions(dims = arr1)
        d2 = Dimensions(dims = arr2)
        d3 = Dimensions(dims = arr3)

        d1_1 = d1 ** 1
        d1_2 = d1 ** 2
        d1__10 = d1 ** (-10)
        d2_1 = d2 ** 1
        d2_2 = d2 ** 2
        d2__10 = d2 ** (-10)
        d3_1 = d3 ** 1
        d3_2 = d3 ** 2
        d3__10 = d3 ** (-10)

        for i in range(7):
            self.assertEqual(d1_1.dims[i], arr1[i])
            self.assertEqual(d1_2.dims[i], arr1[i]*2)
            self.assertEqual(d1__10.dims[i], arr1[i]*(-10))
            self.assertEqual(d2_1.dims[i], arr2[i])
            self.assertEqual(d2_2.dims[i], arr2[i]*2)
            self.assertEqual(d2__10.dims[i], arr2[i]*(-10))
            self.assertEqual(d3_1.dims[i], arr3[i])
            self.assertEqual(d3_2.dims[i], arr3[i]*2)
            self.assertEqual(d3__10.dims[i], arr3[i]*(-10))

    def test_dimensions_eq(self):
        d1 = Dimensions(dims = [1,1,1,0,1,1,1])
        d2 = Dimensions(dims = [1,1,1,0,1,1,1])
        d3 = Dimensions(dims = [])
        d4 = Dimensions(dims = [0,0,0,0,0,0,0])
        self.assertTrue(d1 == d2)
        self.assertTrue(d3 == d4)
        
    def test_dimensions_neq(self):
        d1 = Dimensions(dims = [1,0,0,0,0,0,0])
        d2 = Dimensions(dims = [0,1,1,0,1,0,0])
        d3 = Dimensions(dims = [-1,10,5,2,0,-4,0])
        d4 = Dimensions(dims = [1,1,1,0,1,1,1])
        d5 = Dimensions(dims = [1,1,1,0,1,1,1])
        d6 = Dimensions(dims = [])
        d7 = Dimensions(dims = [0,0,0,0,0,0,0])

        self.assertTrue(d1 != d2)
        self.assertTrue(d1 != d3)
        self.assertTrue(d1 != d4)
        self.assertTrue(d1 != d5)
        self.assertTrue(d1 != d6)
        self.assertTrue(d1 != d7)
        
        self.assertTrue(d2 != d3)
        self.assertTrue(d2 != d4)
        self.assertTrue(d2 != d5)
        self.assertTrue(d2 != d6)
        self.assertTrue(d2 != d7)
        
        self.assertTrue(d3 != d4)
        self.assertTrue(d3 != d5)
        self.assertTrue(d3 != d6)
        self.assertTrue(d3 != d7)
        
        self.assertFalse(d4 != d5)
        self.assertTrue(d4 != d6)
        self.assertTrue(d4 != d7)
        
        self.assertTrue(d5 != d6)
        self.assertTrue(d5 != d7)
        
        self.assertFalse(d6 != d7)

if __name__ == '__main__':
    unittest.main()
