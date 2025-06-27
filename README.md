# MIRAGE++: Mirror Descent Entropy-Regularized Linear Regression with Geometric Extensions

## Overview
MIRAGE++ is a modular, research-grade linear regression framework for quant finance and machine learning. It combines:
- Entropy regularization (for smooth, diversified weights)
- Mirror descent optimization (KL and Riemannian geometry)
- Applications to portfolio allocation and alpha signal combination
- Extensible to manifold and SPD matrix learning

## Theoretical Foundations

### Mirror Descent
Mirror descent is an optimization framework that generalizes gradient descent by mapping updates into dual geometric spaces using Bregman divergences. In MIRAGE++, this allows for flexible regularization and geometric extensions.

### Entropy Regularization
Entropy regularization encourages smooth, diversified solutions by penalizing low-entropy (peaked) weight vectors. The Shannon entropy is computed as:

$$ H(x) = -\sum_i x_i \log x_i $$

### KL Divergence
The Kullback-Leibler (KL) divergence is used as a Bregman divergence for mirror descent and as a loss for probability distributions:

$$ D_{KL}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i} $$

### Probability Simplex and Projections
Many MIRAGE++ modules operate on the probability simplex (vectors with non-negative entries summing to 1). Utility functions include normalization, softmax, entropy, KL divergence, and projection onto the simplex.

### Geometric Modeling
The framework is extensible to Riemannian geometry and symmetric positive definite (SPD) matrix learning, enabling advanced applications in finance and machine learning.

### References
- Bubeck, S. (2015). Convex Optimization: Algorithms and Complexity.
- Wang & Carreira-Perpinan (2013). Projection onto the probability simplex.
- Cover & Thomas (2006). Elements of Information Theory.

## Structure
```
mirror_linear_regression/
    __init__.py
    core.py           # Core model logic
    loss.py           # Entropy-regularized loss
    optim.py          # Mirror descent update
    bregman.py        # Bregman divergences (KL, etc.)
    utils.py          # Helpers
    geometry.py       # Riemannian/mirror descent extensions
    portfolio_alloc.py# Portfolio allocation application
    finance_alpha.py  # Alpha signal combination

demo.py              # Example usage and visualization
```

## Usage
See `demo.py` for portfolio allocation and signal combination examples. All weights are normalized and interpretable as distributions.

## Extensions
- Riemannian mirror descent (geometry.py)
- SPD matrix learning (covariance estimation)
- Plug-and-play for new Bregman divergences

## License
MIT
