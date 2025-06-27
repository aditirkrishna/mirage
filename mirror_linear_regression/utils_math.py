"""
utils_math.py

Utility functions for linear algebra and probability/statistics.
"""
import numpy as np

def normalize(x):
    """Normalize a vector to sum to 1 (probability simplex)."""
    x = np.asarray(x)
    s = np.sum(x)
    if s == 0:
        raise ValueError("Sum of vector is zero, cannot normalize.")
    return x / s

def softmax(x):
    """Softmax transformation."""
    x = np.asarray(x)
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def entropy(x):
    """Shannon entropy of a probability vector x."""
    x = np.asarray(x)
    x = x[x > 0]  # filter zeros to avoid log(0)
    return -np.sum(x * np.log(x))

def project_simplex(v):
    """Project a vector v onto the probability simplex."""
    # Implements the algorithm from [Wang & Carreira-Perpinan, 2013]
    v = np.asarray(v)
    if np.sum(v) == 1 and np.all(v >= 0):
        return v
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

# Add more utility functions as needed

def kl_divergence(p, q):
    """Kullback-Leibler divergence D_KL(p || q) for probability vectors p, q."""
    p = np.asarray(p)
    q = np.asarray(q)
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

