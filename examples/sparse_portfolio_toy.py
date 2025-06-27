import numpy as np

def toy_sparse_portfolio(T=30, N=100, noise=0.01, seed=1):
    np.random.seed(seed)
    returns = np.random.randn(T, N) * 0.02 + 0.001
    return returns
