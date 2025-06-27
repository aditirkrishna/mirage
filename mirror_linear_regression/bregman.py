# bregman.py
import numpy as np

def kl_divergence(p, q, epsilon=1e-8):
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))

# Placeholder for other Bregman divergences (e.g., Itakura-Saito, Mahalanobis)
