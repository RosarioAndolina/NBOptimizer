import numpy as np
from numba import njit
from .globals import DTYPE

@njit
def jacobian_finite_diff_acc_2(residual, p, y, h=DTYPE(1e-5)):
    """
    Compute the Jacobian using 2nd-order finite differences.
    """
    n = len(p)
    jac = np.zeros((len(y), n), dtype=DTYPE)
    for i in range(n):
        p_plus = np.copy(p)
        p_minus = np.copy(p)
        p_plus[i] += h
        p_minus[i] -= h
        jac[:, i] = (DTYPE(-0.5) * residual(y, p_minus) + DTYPE(0.5) * residual(y, p_plus)) / h
    return jac

@njit
def jacobian_finite_diff_acc_4(residual, p, y, h=DTYPE(1e-5)):
    """
    Compute the Jacobian using 4th-order finite differences.
    """
    n = len(p)
    jac = np.zeros((len(y), n), dtype=DTYPE)
    for i in range(n):
        p_plus1 = np.copy(p)
        p_plus2 = np.copy(p)
        p_minus1 = np.copy(p)
        p_minus2 = np.copy(p)
        p_plus1[i] += h
        p_plus2[i] += DTYPE(2) * h
        p_minus1[i] -= h
        p_minus2[i] -= DTYPE(2) * h
        jac[:, i] = (
            DTYPE(1/12) * residual(y, p_minus2) -
            DTYPE(2/3) * residual(y, p_minus1) +
            DTYPE(2/3) * residual(y, p_plus1) -
            DTYPE(1/12) * residual(y, p_plus2)
        ) / h
    return jac

@njit
def jacobian_finite_diff_acc_6(residual, p, y, h=DTYPE(1e-5)):
    """
    Compute the Jacobian using 6th-order finite differences.
    """
    n = len(p)
    jac = np.zeros((len(y), n), dtype=DTYPE)
    for i in range(n):
        p_plus1 = np.copy(p)
        p_plus2 = np.copy(p)
        p_plus3 = np.copy(p)
        p_minus1 = np.copy(p)
        p_minus2 = np.copy(p)
        p_minus3 = np.copy(p)
        p_plus1[i] += h
        p_plus2[i] += DTYPE(2) * h
        p_plus3[i] += DTYPE(3) * h
        p_minus1[i] -= h
        p_minus2[i] -= DTYPE(2) * h
        p_minus3[i] -= DTYPE(3) * h
        jac[:, i] = (
            DTYPE(-1/60) * residual(y, p_minus3) +
            DTYPE(3/20) * residual(y, p_minus2) -
            DTYPE(3/4) * residual(y, p_minus1) +
            DTYPE(3/4) * residual(y, p_plus1) -
            DTYPE(3/20) * residual(y, p_plus2) +
            DTYPE(1/60) * residual(y, p_plus3)
        ) / h
    return jac

