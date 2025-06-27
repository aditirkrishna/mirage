import numpy as np

def toy_alpha_signal_combination(n_samples=200, n_signals=10, noise=0.01, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_signals)
    true_weights = np.random.dirichlet(np.ones(n_signals))
    y = X @ true_weights + noise * np.random.randn(n_samples)
    return X, y, true_weights
