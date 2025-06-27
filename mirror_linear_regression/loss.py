# loss.py
import numpy as np

def entropy(theta: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Compute Shannon entropy of a probability vector theta.
    Args:
        theta: Probability vector.
        epsilon: Small value to avoid log(0).
    Returns:
        Entropy value (float).
    """
    theta = np.clip(theta, epsilon, 1.0)
    return -np.sum(theta * np.log(theta))

def loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray, lam: float) -> float:
    """
    Compute entropy-regularized mean squared error loss.
    Args:
        X: Feature matrix.
        y: Target vector.
        theta: Model parameters.
        lam: Entropy regularization strength.
    Returns:
        Regularized loss value (float).
    """
    preds = X @ theta
    mse = np.mean((preds - y)**2)
    ent = entropy(theta)
    return mse + lam * ent
