# core.py
# Core MIRAGE++ model: Entropy-regularized linear regression with mirror descent

import numpy as np
from .loss import loss
from .optim import mirror_descent_step

class MirrorLinearRegression:
    """
    Entropy-regularized linear regression with mirror descent optimization.
    """
    def __init__(self, learning_rate: float = 0.1, n_iters: int = 1000, lam: float = 0.01, tol: float = 1e-6, verbose: bool = False):
        """
        Args:
            learning_rate: Step size for mirror descent.
            n_iters: Maximum number of iterations.
            lam: Entropy regularization strength.
            tol: Early stopping tolerance on loss improvement.
            verbose: Print progress if True.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lam = lam
        self.tol = tol
        self.verbose = verbose
        self.theta = None
        self.loss_history = []

    def _gradient(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        preds = X @ theta
        grad_mse = 2 * X.T @ (preds - y) / len(y)
        grad_entropy = -1 - np.log(np.clip(theta, 1e-8, 1.0))
        return grad_mse + self.lam * grad_entropy

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to data X, y using mirror descent.
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        m, n = X.shape
        self.theta = np.ones(n) / n
        prev_loss = float('inf')
        for i in range(self.n_iters):
            grad = self._gradient(X, y, self.theta)
            self.theta = mirror_descent_step(grad, self.theta, self.learning_rate)
            current_loss = loss(X, y, self.theta, self.lam)
            self.loss_history.append(current_loss)
            if self.verbose and i % 100 == 0:
                print(f"Iter {i}: loss={current_loss:.6f}")
            if abs(prev_loss - current_loss) < self.tol:
                if self.verbose:
                    print(f"Early stopping at iter {i}: loss improvement < tol")
                break
            prev_loss = current_loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for input X.
        """
        if self.theta is None:
            raise ValueError("Model is not fitted yet.")
        return X @ self.theta
