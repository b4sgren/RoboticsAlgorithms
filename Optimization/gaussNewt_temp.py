import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt

class RangeResidual:
    def __init__(self, range, position, cov):
        self._range = range
        self._position = position
        self._info = 1/cov

    def __call__(self, lm):
        r_hat = np.linalg.norm(lm - self._position)
        return (r_hat - self._range)**2 * self._info

    def getInfo(self):
        return self._info

    def jacobian(self, lm):
        dx = lm - self._position
        r_hat = np.linalg.norm(dx)
        jac = 2 * self._info * (r_hat - self._range) * dx/r_hat
        return jac

    def size(self):
        return 1

class GaussNewton:
    def __init__(self):
        self._vars = {}
        self._residuals = {}
        self._var_sizes = {}

    def addResidual(self, var_id, var, residual):
        if var_id not in self._vars:
            self._vars[var_id] = var
            self._var_sizes[var_id] = var.size
        self._residuals[residual] = var_id

    def optimize(self):
        residual_size = np.sum([r.size() for r in self._residuals.keys()])
        var_size = np.sum(v for v in self._var_sizes.values())
        dx = np.ones(var_size) * 1e6
        iter = 0
        y = 3000

        # Note that this implementation isn't generic
        while np.max(np.abs(dx)) > 1e-3:
            v = np.array([res(self._vars[i]) for res, i in self._residuals.items()])
            R_inv = spl.block_diag(*[res.getInfo() \
                    for res in self._residuals.keys()])
            J = np.zeros((residual_size, var_size))
            J_vec = [res.jacobian(self._vars[val]) \
                    for res, val in self._residuals.items()]
            for i, res in enumerate(self._residuals.keys()):
                J[i] = J_vec[i]

            b = J.T @ R_inv @ v
            A = J.T @ R_inv @ J + y * np.eye(2)
            # if (iter == 10):
            #     debug = 1
            if np.linalg.det(A) == 0:
                break

            dx = np.linalg.solve(A,b) # Normally done via cholesky

            # How to generically update
            self._vars[0] += dx
            iter += 1

        print("Iters: ", iter)
        return np.array([v for v in self._vars.values()]).squeeze()

if __name__=="__main__":
    # Generate Data
    num_pts = 3

    lm = np.random.uniform(-100, 100, size=2)
    print(lm)
    true_pos = [np.random.uniform(-100, 100, size=2) for i in range(num_pts)]
    true_ranges = [np.linalg.norm(lm - p) for p in true_pos]

    # Add Noise
    range_cov = 1e-2
    pos_cov = np.diag([1e-1, 1e-1])
    ranges = [r + np.random.normal(0, np.sqrt(range_cov)) for r in true_ranges]

    residuals = [RangeResidual(r, p, range_cov) for r,p in zip(ranges, true_pos)]
    lm_init = lm + np.array([10, -10])

    optimizer = GaussNewton()
    for res in residuals:
        optimizer.addResidual(0, lm, res)

    lm_est = optimizer.optimize()
    print(lm_est)
