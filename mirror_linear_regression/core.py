# core.py
# Core MIRAGE++ model: Entropy-regularized linear regression with mirror descent

import numpy as np
from .loss import loss
from .optim import mirror_descent_step

class MirrorLinearRegression:
    def __init__(self, learning_rate=0.1, n_iters=1000, lam=0.01):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lam = lam
        self.theta = None
        self.loss_history = []

    def _gradient(self, X, y, theta):
        preds = X @ theta
        grad_mse = 2 * X.T @ (preds - y) / len(y)
        grad_entropy = -1 - np.log(np.clip(theta, 1e-8, 1.0))
        return grad_mse + self.lam * grad_entropy

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.ones(n) / n
        for _ in range(self.n_iters):
            grad = self._gradient(X, y, self.theta)
            self.theta = mirror_descent_step(grad, self.theta, self.learning_rate)
            current_loss = loss(X, y, self.theta, self.lam)
            self.loss_history.append(current_loss)

    def predict(self, X):
        return X @ self.theta
