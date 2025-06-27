# portfolio_alloc.py
import numpy as np
from .core import MirrorLinearRegression

class PortfolioAllocator:
    def __init__(self, learning_rate=0.1, n_iters=1000, lam=0.05):
        self.model = MirrorLinearRegression(learning_rate, n_iters, lam)
        self.theta = None

    def fit(self, returns, target_return=None):
        # returns: T x N matrix (T periods, N assets)
        # target_return: optional, not used in basic version
        X = returns
        y = np.mean(returns, axis=1)  # Use mean return as target
        self.model.fit(X, y)
        self.theta = self.model.theta
        return self

    def get_weights(self):
        return self.theta
