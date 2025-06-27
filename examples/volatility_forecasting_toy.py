import numpy as np

def toy_volatility_forecasting(T=200, n_features=6, seed=7):
    np.random.seed(seed)
    X = np.random.randn(T, n_features)
    true_coefs = np.random.uniform(0.1, 0.5, size=n_features)
    vol = np.abs(X @ true_coefs + 0.1 * np.random.randn(T))
    return X, vol, true_coefs
