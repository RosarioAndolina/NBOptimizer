import numpy as np
from numba import njit
from .globals import DTYPE, ARGUMENTS, MODEL
from .jacobian import jacobian_finite_diff_acc_2, jacobian_finite_diff_acc_4, jacobian_finite_diff_acc_6
from .optimizers import levenberg_marquardt, levenberg_marquardt_improved, gauss_newton, gauss_newton_no_cost

optimizer_dict = {
    'LM': levenberg_marquardt,
    'LMI': levenberg_marquardt_improved,
    'GN': gauss_newton,
    'GN_NC': gauss_newton_no_cost
}

class NBOptimizer:
    def __init__(self, model, p0, args,
                 optimizer = 'LM',
                 accuracy = 2,
                 bounds=None,
                 max_iter=100,
                 tol=DTYPE(1e-6),
                 nu_inc=DTYPE(10),
                 nu_dec=DTYPE(0.1),
                 lambda0=DTYPE(1E-3),
                 tol_bound = DTYPE(1E-10)):

        self.model = model
        self.args = args
        self.accuracy = np.uint8(accuracy)
        self.bounds = bounds
        self.max_iter = max_iter
        self.tol = tol
        self.nu_inc = nu_inc
        self.nu_dec = nu_dec
        self.lambda0 = lambda0
        self.tol_bound = tol_bound
        self.params = p0
        self.residuals = None
        self.jacobian = None
        self.set_optimizer(optimizer)
        self.setup_optimization()

    def set_optimizer(self, optimizer = None):
        if optimizer is not None:
            self.optimizer = optimizer_dict.get(optimizer)
            if not self.optimizer:
                raise ValueError(f'Unknown optimizer {optimizer}')

        if self.bounds is not None:
            self.bounds = DTYPE(self.bounds)

        if optimizer == 'LM':
            self.opt_kwargs = {
                'bounds': self.bounds,
                'max_iter': np.uint16(self.max_iter),
                'tol': DTYPE(self.tol),
                'nu_inc': DTYPE(self.nu_inc),
                'nu_dec': DTYPE(self.nu_dec),
                'lambda0': DTYPE(self.lambda0)
            }

        elif optimizer == 'LMI':
            self.opt_kwargs = {
                'bounds': self.bounds,
                'max_iter': np.uint16(self.max_iter),
                'tol': DTYPE(self.tol),
                'nu_inc': DTYPE(self.nu_inc),
                'nu_dec': DTYPE(self.nu_dec),
                'lambda0': DTYPE(self.lambda0),
                'tol_bound': DTYPE(self.tol_bound)
            }

        elif optimizer == 'GN':
            self.opt_kwargs = {
                'bounds': self.bounds,
                'max_iter': np.uint16(self.max_iter),
                'tol': DTYPE(self.tol),
                'tol_bound': DTYPE(self.tol_bound)
            }

        elif optimizer == 'GN_NC':
            self.opt_kwargs = {
                'bounds': self.bounds,
                'max_iter': np.uint16(self.max_iter),
                'tol': DTYPE(self.tol),
                'tol_bound': DTYPE(self.tol_bound)
            }

        return self

    def set_dtype(self, dtype):
        global DTYPE
        DTYPE = dtype
        self.setup_optimization()
        return self

    def get_model(self):
        global ARGUMENTS, MODEL
        ARGUMENTS = self.args
        MODEL = self.model
        @njit
        def _model(params):
            return MODEL(ARGUMENTS, params)
        return _model

    def get_residuals(self):
        _ = self.get_model()
        @njit
        def _residuals(y_obs, params):
            y = MODEL(ARGUMENTS, params)
            return y_obs - y
        return _residuals

    def get_jacobian_function(self, accuracy = None):
        if accuracy is not None:
            self.accuracy = np.uint8(accuracy)

        if DTYPE == np.float64:
            epsilon = DTYPE(1e-16)
        elif DTYPE == np.float32:
            epsilon = DTYPE(1e-7)
        else:
            raise ValueError("Unsupported dtype. Use float32 or float64.")

        # Calcola h in base all'accuratezza
        h = epsilon ** (DTYPE(1)/self.accuracy)
        print(f'h {h:.2e} {h.dtype}')

        # Seleziona la funzione Jacobiana in base all'accuratezza
        if self.accuracy == 2:
            @njit
            def jacobian(residual, y, p):
                return jacobian_finite_diff_acc_2(residual, p, y, h)
            return jacobian
        elif self.accuracy == 4:
            @njit
            def jacobian(residual, y, p):
                return jacobian_finite_diff_acc_4(residual, p, y, h)
            return jacobian
        elif self.accuracy == 6:
            @njit
            def jacobian(residual, y, p):
                return jacobian_finite_diff_acc_6(residual, p, y, h)
            return jacobian
        else:
            raise ValueError("Unsupported accuracy. Choose 2, 4 or 6.")

    def setup_optimization(self):
        self.set_optimizer()
        self.residuals = self.get_residuals()
        self.jacobian = self.get_jacobian_function()

    def optimize(self, y_obs):
        return self.optimizer(self.residuals, self.params, self.jacobian, DTYPE(y_obs), **self.opt_kwargs)
