"""
NBOptimizer - A Python package for numerical optimization using Numba.

This package provides efficient implementations of the Levenberg-Marquardt and Gauss-Newton
algorithms for solving nonlinear least-squares problems. It leverages Numba's just-in-time (JIT)
compilation for high performance.

Key Features:
- Levenberg-Marquardt algorithm with and without improved bound handling.
- Gauss-Newton algorithm with and without cost calculation.
- Jacobian computation using finite differences (2nd, 4th, and 6th order).
- Support for parameter bounds and custom data types.

Example Usage:
    from nboptimizer import NBOptimizer
    import numpy as np
    from numba import njit

    @njit
    def model(args, p):
        x = args[0]
        return p[0] * x + p[1]

    p0 = np.array([1.0, 1.0], dtype=np.float32)
    bounds = np.array([[2, 3], [3, 5]], dtype=np.float32)
    args = np.array([np.linspace(0, 10, 100)], dtype=np.float32)
    y_obs = 2.5 * args[0] + 3.8 + np.random.normal(0, 0.1, 100).astype(np.float32)

    optimizer = NBOptimizer(model, p0, args, bounds=bounds.T, optimizer='LMI', accuracy=6)
    p_opt = optimizer.optimize(y_obs)
    print("Optimized parameters:", p_opt)
"""

# Import the core class and functions
from .core import NBOptimizer
from .optimizers import (
    levenberg_marquardt,
    levenberg_marquardt_improved,
    gauss_newton,
    gauss_newton_no_cost,
)
from .jacobian import (
    jacobian_finite_diff_acc_2,
    jacobian_finite_diff_acc_4,
    jacobian_finite_diff_acc_6,
)

# Expose the main class and functions to the user
__all__ = [
    "NBOptimizer",
    "levenberg_marquardt",
    "levenberg_marquardt_improved",
    "gauss_newton",
    "gauss_newton_no_cost",
    "jacobian_finite_diff_acc_2",
    "jacobian_finite_diff_acc_4",
    "jacobian_finite_diff_acc_6",
]

# Set the default data type
from .globals import DTYPE

# Package metadata
__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

