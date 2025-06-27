# geometry.py
import numpy as np

# Riemannian (natural) gradient step on the probability simplex

def riemannian_mirror_step(grad, theta_t, eta):
    # Fisher-Rao geometry: exponentiated gradient, then normalize
    theta_new = theta_t * np.exp(-eta * grad)
    theta_new = np.clip(theta_new, 1e-12, None)
    return theta_new / np.sum(theta_new)

# SPD manifold extension placeholder (for covariance matrices)
def spd_riemannian_step(grad, Sigma_t, eta):
    # Placeholder: use matrix exponential for SPD update
    # For now, just a stub
    return Sigma_t  # To be implemented
