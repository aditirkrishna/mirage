import numpy as np

def toy_option_surface(n_points=100, seed=17):
    np.random.seed(seed)
    strikes = np.linspace(80, 120, n_points)
    maturities = np.linspace(0.1, 2.0, n_points)
    surface = 0.2 + 0.1 * np.exp(-((strikes-100)**2)/200) + 0.05 * maturities
    noise = 0.02 * np.random.randn(n_points)
    observed = surface + noise
    return strikes, maturities, observed, surface
