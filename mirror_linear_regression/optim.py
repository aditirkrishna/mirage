# optim.py
import numpy as np

def mirror_descent_step(grad, theta_t, eta):
    # Exponentiated gradient descent update (KL-mirror)
    theta_new = theta_t * np.exp(-eta * grad)
    theta_new = np.clip(theta_new, 1e-12, None)  # Ensure positivity
    return theta_new / np.sum(theta_new)  # Normalize to simplex
