import numpy as np

def toy_factor_exposures(n_stocks=50, n_factors=8, seed=11):
    np.random.seed(seed)
    F = np.random.randn(n_stocks, n_factors)
    exposures = np.random.dirichlet(np.ones(n_factors), size=n_stocks)
    return F, exposures
