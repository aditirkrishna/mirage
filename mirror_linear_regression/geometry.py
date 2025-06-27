# geometry.py
import numpy as np
from scipy.linalg import expm, sqrtm

# Riemannian (natural) gradient step on the probability simplex

def riemannian_mirror_step(grad: np.ndarray, theta_t: np.ndarray, eta: float) -> np.ndarray:
    """
    Fisher-Rao (Riemannian) mirror descent step for probability simplex.
    Args:
        grad: Gradient vector.
        theta_t: Current parameter vector (on simplex).
        eta: Learning rate.
    Returns:
        Updated parameter vector on the simplex.
    """
    theta_new = theta_t * np.exp(-eta * grad)
    theta_new = np.clip(theta_new, 1e-12, None)
    return theta_new / np.sum(theta_new)

# SPD manifold extension for covariance matrices

def spd_riemannian_step(grad: np.ndarray, Sigma_t: np.ndarray, eta: float) -> np.ndarray:
    """
    Riemannian gradient step on SPD manifold (covariance matrices).
    Args:
        grad: Gradient matrix (same shape as Sigma_t).
        Sigma_t: Current SPD matrix.
        eta: Learning rate.
    Returns:
        Updated SPD matrix.
    """
    # Matrix exponential update: Sigma_{t+1} = expm(-eta * grad) @ Sigma_t @ expm(-eta * grad).T
    expG = expm(-eta * grad)
    Sigma_new = expG @ Sigma_t @ expG.T
    return project_to_spd(Sigma_new)


def project_to_spd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Project a matrix onto the SPD cone by symmetrization and eigenvalue thresholding.
    Args:
        A: Input matrix.
        eps: Minimum eigenvalue threshold.
    Returns:
        Symmetric positive definite matrix.
    """
    A = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
