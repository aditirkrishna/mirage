# bregman.py
import numpy as np

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Compute Kullback-Leibler divergence D_KL(p || q) for probability vectors p, q.
    Args:
        p: First probability vector.
        q: Second probability vector.
        epsilon: Small value to avoid log(0).
    Returns:
        KL divergence value (float).
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))

# TODO: Implement other Bregman divergences (e.g., Itakura-Saito, Mahalanobis)
