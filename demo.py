# demo.py
# Example usage of MIRAGE++ for portfolio allocation and alpha signal combination
import numpy as np
from mirror_linear_regression.portfolio_alloc import PortfolioAllocator
from mirror_linear_regression.finance_alpha import AlphaSignalCombiner
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Portfolio Allocation Demo ---
T, N = 100, 10  # 100 days, 10 assets
returns = np.random.randn(T, N) * 0.01 + 0.0005  # Simulated daily returns

allocator = PortfolioAllocator(learning_rate=0.2, n_iters=500, lam=0.1)
allocator.fit(returns)
weights = allocator.get_weights()
print("Portfolio Weights:", weights)
print("Sum of Weights:", np.sum(weights))

# --- Alpha Signal Combination Demo ---
X = np.random.randn(T, N)  # 10 signals
true_theta = np.random.dirichlet(np.ones(N))
y = X @ true_theta + np.random.randn(T) * 0.05

combiner = AlphaSignalCombiner(learning_rate=0.2, n_iters=500, lam=0.1)
combiner.fit(X, y)
signal_weights = combiner.get_signal_weights()
print("Signal Weights:", signal_weights)

# --- Visualization ---
plt.figure(figsize=(10,4))
plt.bar(range(N), weights, alpha=0.6, label='Portfolio Weights')
plt.bar(range(N), signal_weights, alpha=0.6, label='Signal Weights')
plt.xlabel('Asset/Signal Index')
plt.ylabel('Weight')
plt.title('MIRAGE++ Weights')
plt.legend()
plt.show()
