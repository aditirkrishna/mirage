import numpy as np

def toy_ensemble_forecasting(n_models=15, n_obs=200, seed=21):
    np.random.seed(seed)
    signals = np.random.randn(n_obs, n_models)
    true_weights = np.random.dirichlet(np.ones(n_models))
    y = signals @ true_weights + 0.05 * np.random.randn(n_obs)
    return signals, y, true_weights
