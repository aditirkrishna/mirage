"""
Toy dataset generators for MIRAGE++ quant finance use cases.
Each function returns (X, y) or (returns, features) as appropriate.
"""
import numpy as np

def toy_alpha_signal_combination(n_samples=200, n_signals=10, noise=0.01, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_signals)
    true_weights = np.random.dirichlet(np.ones(n_signals))
    y = X @ true_weights + noise * np.random.randn(n_samples)
    return X, y, true_weights

def toy_sparse_portfolio(T=30, N=100, noise=0.01, seed=1):
    np.random.seed(seed)
    returns = np.random.randn(T, N) * 0.02 + 0.001
    return returns

def toy_volatility_forecasting(T=200, n_features=6, seed=7):
    np.random.seed(seed)
    X = np.random.randn(T, n_features)
    # True volatility is a smooth function of features
    true_coefs = np.random.uniform(0.1, 0.5, size=n_features)
    vol = np.abs(X @ true_coefs + 0.1 * np.random.randn(T))
    return X, vol, true_coefs

def toy_factor_exposures(n_stocks=50, n_factors=8, seed=11):
    np.random.seed(seed)
    F = np.random.randn(n_stocks, n_factors)
    exposures = np.random.dirichlet(np.ones(n_factors), size=n_stocks)
    return F, exposures

def toy_ensemble_forecasting(n_models=15, n_obs=200, seed=21):
    np.random.seed(seed)
    signals = np.random.randn(n_obs, n_models)
    true_weights = np.random.dirichlet(np.ones(n_models))
    y = signals @ true_weights + 0.05 * np.random.randn(n_obs)
    return signals, y, true_weights

def toy_option_surface(n_points=100, seed=17):
    np.random.seed(seed)
    strikes = np.linspace(80, 120, n_points)
    maturities = np.linspace(0.1, 2.0, n_points)
    # True surface: smooth, convex
    surface = 0.2 + 0.1 * np.exp(-((strikes-100)**2)/200) + 0.05 * maturities
    noise = 0.02 * np.random.randn(n_points)
    observed = surface + noise
    return strikes, maturities, observed, surface

def toy_macro_predictive(n_obs=150, n_features=12, seed=31):
    np.random.seed(seed)
    X = np.random.randn(n_obs, n_features)
    true_coefs = np.random.uniform(-0.2, 0.2, size=n_features)
    y = X @ true_coefs + 0.1 * np.random.randn(n_obs)
    return X, y, true_coefs
