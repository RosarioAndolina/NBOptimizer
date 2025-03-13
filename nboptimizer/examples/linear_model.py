"""
Example: Linear Model Optimization

This example demonstrates how to use the NBOptimizer class to fit a linear model to synthetic data.
The model is defined as y = p[0] * x + p[1], where p[0] is the slope and p[1] is the intercept.
"""

import numpy as np
from numba import njit
from nboptimizer import NBOptimizer, DTYPE
import matplotlib.pyplot as plt

# Define the linear model
@njit
def linear_model(args, p):
    """
    Linear model: y = p[0] * x + p[1]
    """
    x = args[0]
    return p[0] * x + p[1]

# Generate synthetic data
np.random.seed(42)
x = np.linspace(0, 10, 100, dtype=DTYPE)
y_true = DTYPE(2.5) * x + DTYPE(3.8)  # True parameters: slope = 2.5, intercept = 3.8
y_obs = y_true + np.random.normal(0, 0.5, size=x.shape).astype(DTYPE)  # Add noise

# Define initial parameters and bounds
p0 = np.array([1.0, 1.0], dtype=DTYPE)  # Initial guess: slope = 1.0, intercept = 1.0
bounds = np.array([[2, 3], [3, 5]], dtype=DTYPE)  # Bounds: slope ∈ [2, 3], intercept ∈ [3, 5]

args = (x,)

# Create optimizer instance
optimizer = NBOptimizer(
    model=linear_model,
    p0=p0,
    args=args,
    bounds=bounds.T,
    optimizer='LMI',  # Use improved Levenberg-Marquardt
    accuracy=6,       # Use 6th-order finite differences for Jacobian
    max_iter=100,     # Maximum number of iterations
    tol=DTYPE(1e-6),  # Tolerance for convergence
)

# Run optimization
p_opt = optimizer.optimize(y_obs)
print("Optimized parameters:", p_opt)
_model = optimizer.get_model()

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, y_obs, '.', label="Observed Data")
plt.plot(x, _model(p_opt), 'r-', label="Fitted Model")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Model Optimization")
plt.legend()
plt.show()