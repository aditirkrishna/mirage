"""
vector_space.py

Core abstractions for vector spaces and inner products.
"""
import numpy as np
from abc import ABC, abstractmethod

class VectorSpace(ABC):
    """Abstract base class for a vector space."""
    @abstractmethod
    def zero(self):
        """Return the zero vector."""
        pass

    @abstractmethod
    def inner(self, x, y):
        """Return the inner product of x and y."""
        pass

    @abstractmethod
    def norm(self, x):
        """Return the norm of x."""
        pass

class EuclideanSpace(VectorSpace):
    """Standard Euclidean vector space abstraction."""
    def zero(self, shape):
        """Return the zero vector of given shape."""
        return np.zeros(shape)

    def inner(self, x, y):
        return np.dot(x, y)

    def norm(self, x):
        return np.linalg.norm(x)

# Example usage:
# space = EuclideanSpace()
# x = np.array([1, 2, 3])
# y = np.array([4, 5, 6])
# print(space.inner(x, y))
# print(space.norm(x))
