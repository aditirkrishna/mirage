# utils.py
import numpy as np

def normalize_simplex(theta):
    theta = np.clip(theta, 1e-12, None)
    return theta / np.sum(theta)

# Add more helpers as needed
