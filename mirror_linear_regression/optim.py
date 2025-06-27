# optim.py
import numpy as np

def mirror_descent_step(grad: np.ndarray, theta_t: np.ndarray, eta: float) -> np.ndarray:
    """
    Perform one mirror descent (exponentiated gradient) step for KL-mirror geometry.
    Args:
        grad: Gradient vector.
        theta_t: Current parameter vector.
        eta: Learning rate.
    Returns:
        Updated parameter vector, projected onto the simplex.
    """
    theta_new = theta_t * np.exp(-eta * grad)
    theta_new = np.clip(theta_new, 1e-12, None)  # Ensure positivity
    return theta_new / np.sum(theta_new)  # Normalize to simplex
