import numpy as np

def toy_macro_predictive(n_obs=150, n_features=12, seed=31):
    np.random.seed(seed)
    X = np.random.randn(n_obs, n_features)
    true_coefs = np.random.uniform(-0.2, 0.2, size=n_features)
    y = X @ true_coefs + 0.1 * np.random.randn(n_obs)
    return X, y, true_coefs
