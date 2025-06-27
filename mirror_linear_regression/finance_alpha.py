# finance_alpha.py
import numpy as np
from .core import MirrorLinearRegression

class AlphaSignalCombiner:
    def __init__(self, learning_rate=0.1, n_iters=1000, lam=0.05):
        self.model = MirrorLinearRegression(learning_rate, n_iters, lam)
        self.theta = None

    def fit(self, X, y):
        # X: T x N matrix of signals, y: T vector of future returns
        self.model.fit(X, y)
        self.theta = self.model.theta
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_signal_weights(self):
        return self.theta
