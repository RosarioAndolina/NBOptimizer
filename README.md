# NBOptimizer

A Python package for numerical optimization using Numba. This package implements the Levenberg-Marquardt and Gauss-Newton algorithms for solving nonlinear least-squares problems. It is designed to be fast and efficient, leveraging Numba's just-in-time (JIT) compilation for performance-critical parts of the code.

---

## Table of Contents
1. [Theoretical Introduction](#theoretical-introduction)
   - [Levenberg-Marquardt Algorithm](#levenberg-marquardt-algorithm)
   - [Gauss-Newton Algorithm](#gauss-newton-algorithm)
   - [Differences Between Algorithms](#differences-between-algorithms)
   - [Jacobian Calculation](#jacobian-calculation)
   - [Parameters and Their Meaning](#parameters-and-their-meaning)
2. [Code Overview](#code-overview)
   - [NBOptimizer Class](#nboptimizer-class)
   - [How to Use the Class](#how-to-use-the-class)
3. [Installation](#installation)
4. [Examples](#examples)

---

## Theoretical Introduction

### Levenberg-Marquardt Algorithm
The Levenberg-Marquardt (LM) algorithm is a widely used method for solving nonlinear least-squares problems. It combines the advantages of the **gradient descent** and **Gauss-Newton** methods. The algorithm adjusts its behavior based on a damping parameter (λ):

- When λ is large, the method behaves like **gradient descent**, taking small, safe steps.
- When λ is small, it behaves like **Gauss-Newton**, taking larger steps toward the minimum.

The update rule for the parameters is given by:

$$
\Delta p = -(J^T J + \lambda I)^{-1} J^T r
$$

Where:
- **J** is the Jacobian matrix of the residuals.
- **r** is the residual vector.
- **λ** is the damping parameter.
- **I** is the identity matrix.

The algorithm dynamically adjusts λ during optimization to ensure convergence.

---

### Gauss-Newton Algorithm
The Gauss-Newton (GN) algorithm is a simplification of the Newton method for least-squares problems. It approximates the Hessian matrix using the Jacobian (JᵀJ) and is particularly effective when the residuals are small.

The update rule for the parameters is:

$$
\Delta p = -(J^T J)^{-1} J^T r
$$

Where:
- **J** is the Jacobian matrix of the residuals.
- **r** is the residual vector.

Unlike LM, GN does not use a damping parameter, which can make it faster but less robust for problems with large residuals or poor initial guesses.

---

### Differences Between Algorithms
1. **Levenberg-Marquardt**:
   - More robust and stable.
   - Suitable for problems with large residuals or poor initial guesses.
   - Slower due to the damping parameter adjustment.

2. **Gauss-Newton**:
   - Faster for small residuals.
   - Less robust for difficult problems.
   - Does not require a damping parameter.

3. **Improved Levenberg-Marquardt**:
   - Adds better handling of parameter bounds.
   - Projects the gradient to ensure constraints are respected.

4. **Gauss-Newton (No Cost)**:
   - A variant of GN that does not compute the cost function at each step.
   - Useful when the cost function is expensive to compute.

---

### Jacobian Calculation
The Jacobian matrix is computed using **finite differences**. The package provides three levels of accuracy:
1. **2nd-Order Finite Differences**: Basic accuracy, fast computation.
2. **4th-Order Finite Differences**: Higher accuracy, moderate computation.
3. **6th-Order Finite Differences**: Highest accuracy, slower computation.

The step size (h) for finite differences is automatically calculated based on the machine epsilon and the desired accuracy.

---

### Parameters and Their Meaning
The following parameters are used to instantiate the `NBOptimizer` class:
- **`model`**: The model function that takes `args` and `p` as inputs.
- **`p0`**: Initial guess for the parameters.
- **`args`**: Input data passed to the model.
- **`bounds`**: Parameter bounds as a 2D array (lower and upper bounds for each parameter).
- **`optimizer`**: The optimization algorithm to use (`'LM'`, `'LMI'`, `'GN'`, or `'GN_NC'`).
- **`accuracy`**: The order of accuracy for Jacobian calculation (2, 4, or 6).
- **`max_iter`**: Maximum number of iterations.
- **`tol`**: Tolerance for convergence.
- **`nu_inc`**, **`nu_dec`**: Factors for adjusting the damping parameter in LM.
- **`lambda0`**: Initial value of the damping parameter in LM.
- **`tol_bound`**: Tolerance for bound constraints.

---

## Code Overview

### NBOptimizer Class
The `NBOptimizer` class is the core of the package. It provides the following methods:
1. **`__init__`**: Initializes the optimizer with the model, parameters, and settings.
2. **`set_optimizer`**: Configures the optimization algorithm.
3. **`set_dtype`**: Sets the data type for computations (e.g., `np.float32` or `np.float64`).
4. **`get_model`**: Returns the compiled wrapper of model function.
5. **`get_residuals`**: Returns the residual function.
6. **`get_jacobian_function`**: Returns the Jacobian function based on the selected accuracy.
7. **`optimize`**: Runs the optimization and returns the optimized parameters.

---

### How to Use the Class
1. Define your model function:
   ```python
   from nboptimizer import NBOptimizer, DTYPE
   # DTYPE = np.float32 by default

   # y = ax + b
   # args is a numpy array with the arguments neded by the model
   # args = np.array([x], dtype = DTYPE) in this example
   @njit
   def model(args, p):
       x = args[0]
       return p[0] * x + p[1]
   ```

2. Define the initial parameters and bounds:
   ```python
   p0 = np.array([1.0, 1.0], dtype=DTYPE)
   bounds = np.array([[2, 3], [3, 5]], dtype=DTYPE)  # Bounds for p[0] and p[1]
   # bounds are defined in this way for semplicity and clarity
   # but use bounds.T as argument in NBOptimizer class
   ```

3. Optimize:
   ```python
   x = np.linspace(0, 10, 100, dtype = DTYPE)
   args = DTYPE([x])
   optimizer = NBOptimizer(model, p0, args, bounds=bounds.T, optimizer='LMI', accuracy=2)
   p_opt = optimizer.optimize(y_obs) # y_obs are observed data
   model_wrapper = optimizer.get_model()
   y_fit = model_wrapper(p_opt)
   ```
Use alwais `DTYPE` to define your array and variables:
```python
foo = DTYPE(1.0)
```
To change the default DTYPE use:
```python
optimizer.set_dtype(np.float64)
```
To change the optimizer with the same instance of NBOptimizer:
```python
optimizer.set_optimizer("GN")
```
To change parameters as instance attribute do this alwais:
```python
optimizer.accuracy = 4 # example
optimizer.setup_optimization()
```

### Installation
To install the package, use pip in the package directory:
   ```bash
   pip install .
   ```
### Examples
See **nboptimizer/examples/**
