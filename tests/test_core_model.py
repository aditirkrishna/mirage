import numpy as np
from mirror_linear_regression.core import MirrorLinearRegression

import unittest

class TestMirrorLinearRegression(unittest.TestCase):
    def test_fit_and_predict(self):
        # Simple linear regression test
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_theta = np.array([0.2, 0.5, 0.3])
        y = X @ true_theta + 0.01 * np.random.randn(100)

        model = MirrorLinearRegression(learning_rate=0.2, n_iters=500, lam=0.05, tol=1e-7, verbose=False)
        model.fit(X, y)
        y_pred = model.predict(X)

        # Assert the shape
        self.assertEqual(y_pred.shape, y.shape)
        # Assert the model learns something reasonable
        mse = np.mean((y_pred - y) ** 2)
        print(f"Test MSE: {mse:.6f}")
        self.assertLess(mse, 0.05)  # Should be a low error for synthetic data

if __name__ == "__main__":
    unittest.main()
