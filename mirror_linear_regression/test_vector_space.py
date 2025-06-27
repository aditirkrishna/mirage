import numpy as np
from vector_space import EuclideanSpace

def test_zero():
    space = EuclideanSpace()
    assert space.zero() == 0.0

def test_inner():
    space = EuclideanSpace()
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    assert space.inner(x, y) == np.dot(x, y)

def test_norm():
    space = EuclideanSpace()
    x = np.array([3, 4])
    assert space.norm(x) == 5.0

if __name__ == "__main__":
    test_zero()
    test_inner()
    test_norm()
    print("All tests passed.")
