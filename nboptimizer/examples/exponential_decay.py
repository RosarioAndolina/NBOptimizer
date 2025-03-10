"""
Example: Multiple Exponential Decay Optimization

This example demonstrates how to use the NBOptimizer class to fit a model of multiple exponential decays
to synthetic data. The model is defined as:
y = A1 * exp(-x / tau1) + A2 * exp(-x / tau2) + A3 * exp(-x / tau3)
"""

import numpy as np
from numba import njit
from nboptimizer import NBOptimizer, DTYPE
import matplotlib.pyplot as plt

# Define the exponential decay model
@njit
def exponential_decay_model(args, p):
    """
    Multiple exponential decay model:
    y = A1 * exp(-x / tau1) + A2 * exp(-x / tau2) + A3 * exp(-x / tau3)
    """
    x = args[0]
    n_exp = len(p) // 2  # Number of exponential terms
    out = np.zeros_like(x, dtype=DTYPE)
    for i in range(n_exp):
        A, tau = p[2 * i], p[2 * i + 1]
        out += A * np.exp(-x / tau)
    return out

# Generate synthetic data
np.random.seed(42)
t = np.linspace(0, 2, 100, dtype=DTYPE)  # Time values

p_true = np.array(
    [5.0, 1.0, # A, tau, first exponential decay
    3.0, 0.5, # A, tau, second exponetial decay
    2.0, 0.2], # A, tau, third exponential decay
    dtype = DTYPE)

args = np.array([t], dtype = DTYPE)
y_true = exponential_decay_model(args, p_true)

y_obs = y_true + DTYPE(0.4) * np.random.normal(size=t.shape).astype(DTYPE)  # Add noise

# Define initial parameters and bounds
p0 = np.array([1.0, 0.1, 1.0, 0.1, 1.0, 0.1], dtype=DTYPE)  # Initial guess
# p0 = np.array([0, 1e-6, 0, 1e-5, 0, 1e-6], dtype = DTYPE) # [A1, tau1, A2, tau2, A3, tau3]

bounds = np.array([
    [0, 10], [0.1, 2],  # Bounds for A1 and tau1
    [0, 10], [0.05, 1],  # Bounds for A2 and tau2
    [0, 10], [0.01, 0.5] # Bounds for A3 and tau3
], dtype=DTYPE)

# Create optimizer instance
optimizer = NBOptimizer(
    model=exponential_decay_model,
    p0=p0,
    args = args,
    bounds=bounds.T,
    optimizer='LMI',  # Use improved Levenberg-Marquardt
    accuracy=2,       # Use 6th-order finite differences for Jacobian
    max_iter=100,     # Maximum number of iterations
    tol=DTYPE(1e-6),  # Tolerance for convergence
)

# Run optimization
p_opt = optimizer.optimize(y_obs)
print("Optimized parameters:", p_opt)
_model = optimizer.get_model()

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, y_true, label="True Signal")
plt.plot(t, y_obs, '.', label="Noisy Data")
plt.plot(t, _model(p_opt), 'r--', label="Fitted Model")
plt.xlabel("Time (t)")
plt.ylabel("y(t)")
plt.title("Multiple Exponential Decay Optimization")
plt.legend()
plt.show()
