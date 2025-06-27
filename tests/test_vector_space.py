import numpy as np
from mirror_linear_regression.vector_space import EuclideanSpace
import unittest

class TestVectorSpace(unittest.TestCase):
    def test_zero(self):
        space = EuclideanSpace()
        z = space.zero(3)
        self.assertTrue(np.allclose(z, np.zeros(3)))

    def test_inner(self):
        space = EuclideanSpace()
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        ip = space.inner(x, y)
        self.assertTrue(np.isclose(ip, 32.0))

    def test_norm(self):
        space = EuclideanSpace()
        x = np.array([3.0, 4.0, 0.0])
        n = space.norm(x)
        self.assertTrue(np.isclose(n, 5.0))

if __name__ == "__main__":
    unittest.main()
