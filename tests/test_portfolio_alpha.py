import numpy as np
from mirror_linear_regression.portfolio_alloc import PortfolioAllocator
from mirror_linear_regression.finance_alpha import AlphaSignalCombiner

import unittest

class TestPortfolioAlpha(unittest.TestCase):
    def test_portfolio_allocator(self):
        np.random.seed(0)
        returns = np.random.randn(120, 4) * 0.02 + 0.001  # T=120, N=4 assets
        allocator = PortfolioAllocator(learning_rate=0.2, n_iters=300, lam=0.1)
        allocator.fit(returns)
        weights = allocator.get_weights()
        print("Portfolio Weights:", weights)
        self.assertEqual(weights.shape, (4,))
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.isclose(np.sum(weights), 1.0))

    def test_alpha_signal_combiner(self):
        np.random.seed(1)
        X = np.random.randn(100, 3)  # T=100, N=3 signals
        y = 0.3 * X[:, 0] + 0.5 * X[:, 1] + 0.2 * X[:, 2] + 0.01 * np.random.randn(100)
        combiner = AlphaSignalCombiner(learning_rate=0.15, n_iters=400, lam=0.08)
        combiner.fit(X, y)
        pred = combiner.predict(X)
        weights = combiner.get_signal_weights()
        print("Alpha Weights:", weights)
        self.assertEqual(weights.shape, (3,))
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.isclose(np.sum(weights), 1.0))
        mse = np.mean((pred - y)**2)
        print(f"Alpha MSE: {mse:.6f}")
        self.assertLess(mse, 0.05)

if __name__ == "__main__":
    unittest.main()
