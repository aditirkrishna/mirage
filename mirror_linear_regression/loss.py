# loss.py
import numpy as np

def entropy(theta, epsilon=1e-8):
    theta = np.clip(theta, epsilon, 1.0)
    return -np.sum(theta * np.log(theta))

def loss(X, y, theta, lam):
    preds = X @ theta
    mse = np.mean((preds - y)**2)
    ent = entropy(theta)
    return mse + lam * ent
