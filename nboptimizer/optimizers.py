import numpy as np
from numba import njit
from .globals import DTYPE

@njit
def levenberg_marquardt(residual_func, p0, jac, y_obs,
                        bounds=None,
                        max_iter=100,
                        tol=DTYPE(1e-6),
                        nu_inc=DTYPE(10),
                        nu_dec=DTYPE(0.1),
                        lambda0=1e-4):

    """
    Levenberg-Marquardt optimization algorithm.
    """

    p = p0.copy()
    # x,y = args
    if bounds is not None:
        lower, upper = bounds
        p = np.clip(p, lower, upper)

    m = len(residual_func(y_obs, p))
    n = len(p)

    lambda_ = lambda0 if lambda0 else DTYPE(1e-3)
    cost = np.sum(residual_func(y_obs, p)**2)
    nu = DTYPE(2.0)

    for _ in range(max_iter):
        r = residual_func(y_obs, p)
        J = jac(residual_func, y_obs, p)

        if bounds is not None:
            grad = J.T @ r
            active_lower = (p <= lower + DTYPE(1e-10)) & (grad > 0)
            active_upper = (p >= upper - DTYPE(1e-10)) & (grad < 0)
            active = active_lower | active_upper
            grad[active] = DTYPE(0.0)
            J = J.copy()
            J[:, active] = DTYPE(0.0)

        JtJ = J.T @ J
        diag = np.diag(JtJ)
        damped = JtJ + lambda_ * np.diag(np.maximum(diag, DTYPE(1e-8)))
        delta = np.linalg.solve(damped, -J.T @ r)

        p_new = p + delta
        if bounds is not None:
            p_new = np.clip(p_new, lower, upper)

        r_new = residual_func(y_obs, p_new)
        cost_new = np.sum(r_new**2)

        if cost_new < cost:
            p = p_new
            cost = cost_new
            lambda_ *= nu_dec
            nu = DTYPE(2.0)
        else:
            lambda_ *= nu_inc
            nu *= DTYPE(2.0)

        if np.linalg.norm(delta) < tol:
            break
    # print(f'niter {_}')
    return p



@njit
def levenberg_marquardt_improved(residual_func, p0, jac, y_obs,
                                 bounds=None,
                                 max_iter=100,
                                 tol=DTYPE(1e-6),
                                 nu_inc=DTYPE(10),
                                 nu_dec=DTYPE(0.1),
                                 lambda0=DTYPE(1e-4),
                                 tol_bound=DTYPE(1e-10)):
    """
    Improved Levenberg-Marquardt optimization algorithm.
    """
    p = p0.copy()
    if bounds is not None:
        lower, upper = bounds
        p = np.clip(p, lower, upper)

    m = len(residual_func(y_obs, p))
    n = len(p)

    lambda_ = lambda0 if lambda0 else DTYPE(1e-3)
    cost = np.sum(residual_func(y_obs, p)**2)
    nu = DTYPE(2.0)

    for _ in range(max_iter):
        r = residual_func(y_obs, p)
        J = jac(residual_func, y_obs, p)

        if bounds is not None:
            grad = J.T @ r
            active_lower = (p <= lower + tol_bound) & (grad > 0)
            active_upper = (p >= upper - tol_bound) & (grad < 0)
            active = active_lower | active_upper

            # Proiezione del gradiente
            grad_proj = grad.copy()
            grad_proj[active_lower] = np.maximum(grad[active_lower], DTYPE(0.0))
            grad_proj[active_upper] = np.minimum(grad[active_upper], DTYPE(0.0))

            # Riduzione della Jacobiana
            J_reduced = J[:, ~active]
            if J_reduced.size > 0:
                JtJ_reduced = J_reduced.T @ J_reduced
                diag_reduced = np.diag(JtJ_reduced)
                damped_reduced = JtJ_reduced + lambda_ * np.diag(np.maximum(diag_reduced, DTYPE(1e-8)))
                delta_reduced = np.linalg.solve(damped_reduced, -J_reduced.T @ r)
                delta = np.zeros_like(p, dtype=DTYPE)
                delta[~active] = delta_reduced
            else:
                delta = np.zeros_like(p, dtype=DTYPE)
        else:
            # Nessun bound, procedi normalmente
            JtJ = J.T @ J
            diag = np.diag(JtJ)
            damped = JtJ + lambda_ * np.diag(np.maximum(diag, DTYPE(1e-8)))
            delta = np.linalg.solve(damped, -J.T @ r)

        # Aggiornamento dei parametri
        p_new = p + delta
        if bounds is not None:
            p_new = np.clip(p_new, lower, upper)

        # Calcolo del nuovo costo
        r_new = residual_func(y_obs, p_new)
        cost_new = np.sum(r_new**2)

        # Aggiornamento dei parametri e regolazione di lambda_
        if cost_new < cost:
            p = p_new
            cost = cost_new
            lambda_ *= nu_dec
            nu = DTYPE(2.0)
        else:
            lambda_ *= nu_inc
            nu *= DTYPE(2.0)

        # Condizione di convergenza
        if np.linalg.norm(delta) < tol:
            break

    return p


@njit
def gauss_newton(residual_func, p0, jac, y_obs,
                 bounds=None,
                 max_iter=100,
                 tol=DTYPE(1e-6),
                 tol_bound=DTYPE(1e-10)):
    """
    Gauss-Newton optimization algorithm.
    """
    p = p0.copy()
    if bounds is not None:
        lower, upper = bounds
        p = np.clip(p, lower, upper)

    m = len(residual_func(y_obs, p))
    n = len(p)

    cost = np.sum(residual_func(y_obs, p)**2)

    for _ in range(max_iter):
        r = residual_func(y_obs, p)
        J = jac(residual_func, y_obs, p)

        if bounds is not None:
            grad = J.T @ r
            active_lower = (p <= lower + tol_bound) & (grad > 0)
            active_upper = (p >= upper - tol_bound) & (grad < 0)
            active = active_lower | active_upper

            # Proiezione del gradiente
            grad_proj = grad.copy()
            grad_proj[active_lower] = np.maximum(grad[active_lower], DTYPE(0.0))
            grad_proj[active_upper] = np.minimum(grad[active_upper], DTYPE(0.0))

            # Riduzione della Jacobiana
            J_reduced = J[:, ~active]
            if J_reduced.size > 0:
                JtJ_reduced = J_reduced.T @ J_reduced
                diag_reduced = np.diag(JtJ_reduced)
                damped_reduced = JtJ_reduced + np.diag(np.maximum(diag_reduced, DTYPE(1e-8)))  # Regolarizzazione adattiva
                delta_reduced = np.linalg.solve(damped_reduced, -J_reduced.T @ r)
                delta = np.zeros_like(p, dtype=DTYPE)
                delta[~active] = delta_reduced
            else:
                delta = np.zeros_like(p, dtype=DTYPE)
        else:
            # Nessun bound, procedi normalmente
            JtJ = J.T @ J
            diag = np.diag(JtJ)
            damped = JtJ + np.diag(np.maximum(diag, DTYPE(1e-8)))  # Regolarizzazione adattiva
            delta = np.linalg.solve(damped, -J.T @ r)

        # Aggiornamento dei parametri
        p_new = p + delta
        if bounds is not None:
            p_new = np.clip(p_new, lower, upper)

        # Calcolo del nuovo costo
        r_new = residual_func(y_obs, p_new)
        cost_new = np.sum(r_new**2)

        # Aggiornamento dei parametri
        if cost_new < cost:
            p = p_new
            cost = cost_new
        else:
            # Se il costo non migliora, interrompi l'iterazione
            break

        # Condizione di convergenza
        if np.linalg.norm(delta) < tol:
            break

    return p


@njit
def gauss_newton_no_cost(residual_func, p0, jac, y_obs,
                         bounds=None,
                         max_iter=100,
                         tol=DTYPE(1e-6),
                         tol_bound=DTYPE(1e-10)):
    """
    Gauss-Newton optimization algorithm without cost calculation.
    """
    p = p0.copy()
    if bounds is not None:
        lower, upper = bounds
        p = np.clip(p, lower, upper)

    n = len(p)

    for _ in range(max_iter):
        r = residual_func(y_obs, p)
        J = jac(residual_func, y_obs, p)

        if bounds is not None:
            grad = J.T @ r
            active_lower = (p <= lower + tol_bound) & (grad > 0)
            active_upper = (p >= upper - tol_bound) & (grad < 0)
            active = active_lower | active_upper

            # Proiezione del gradiente
            grad_proj = grad.copy()
            grad_proj[active_lower] = np.maximum(grad[active_lower], DTYPE(0.0))
            grad_proj[active_upper] = np.minimum(grad[active_upper], DTYPE(0.0))

            # Riduzione della Jacobiana
            J_reduced = J[:, ~active]
            if J_reduced.size > 0:
                JtJ_reduced = J_reduced.T @ J_reduced
                diag_reduced = np.diag(JtJ_reduced)
                damped_reduced = JtJ_reduced + np.diag(np.maximum(diag_reduced, DTYPE(1e-8)))  # Regolarizzazione adattiva
                delta_reduced = np.linalg.solve(damped_reduced, -J_reduced.T @ r)
                delta = np.zeros_like(p, dtype=DTYPE)
                delta[~active] = delta_reduced
            else:
                delta = np.zeros_like(p, dtype=DTYPE)
        else:
            # Nessun bound, procedi normalmente
            JtJ = J.T @ J
            diag = np.diag(JtJ)
            damped = JtJ + np.diag(np.maximum(diag, DTYPE(1e-8)))  # Regolarizzazione adattiva
            delta = np.linalg.solve(damped, -J.T @ r)

        # Aggiornamento dei parametri
        p_new = p + delta
        if bounds is not None:
            p_new = np.clip(p_new, lower, upper)

        # Condizione di convergenza basata sulla norma del passo
        if np.linalg.norm(delta) < tol:
            break

        # Aggiornamento dei parametri
        p = p_new

    return p
