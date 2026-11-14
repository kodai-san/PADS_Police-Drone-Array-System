"""
MPC QP solver — direct port of solveMPC_QP.m.

Minimises:
  sum_k || x_k - xdes_k ||^2_Q  +  || U ||^2_R  +  w_du || ΔU ||^2

Subject to:
  -amax <= u_k <= amax   (component-wise)

Uses scipy quadprog-equivalent or OSQP when available.
"""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import Tuple, Optional

try:
    import osqp
    from scipy import sparse
    _HAS_OSQP = True
except ImportError:
    _HAS_OSQP = False


# ── Prediction matrices ────────────────────────────────────────────────
def build_prediction_matrices(A: np.ndarray, B: np.ndarray, N: int):
    """Port of buildPredictionMatrices.m
    Returns Phi [(N+1)*nx, nx], Gamma [(N+1)*nx, N*nu].
    """
    nx = A.shape[0]
    nu = B.shape[1]
    Phi = np.zeros(((N + 1) * nx, nx))
    Gamma = np.zeros(((N + 1) * nx, N * nu))
    Phi[0:nx, :] = np.eye(nx)
    A_pow = np.eye(nx)
    for i in range(1, N + 1):
        A_pow = A_pow @ A
        r = slice(i * nx, (i + 1) * nx)
        Phi[r, :] = A_pow
        A_pow_j = np.eye(nx)
        for j in range(i, 0, -1):
            c = slice((j - 1) * nu, j * nu)
            Gamma[r, c] = A_pow_j @ B
            A_pow_j = A_pow_j @ A
    return Phi, Gamma


def diff_matrix_2d(N: int) -> np.ndarray:
    """Port of diffMatrix2D.m — first-difference for 2-D input sequence."""
    nu = 2
    Du = np.zeros((nu * (N - 1), nu * N))
    for k in range(N - 1):
        rows = slice(2 * k, 2 * k + 2)
        prev = slice(2 * k, 2 * k + 2)
        curr = slice(2 * (k + 1), 2 * (k + 1) + 2)
        Du[rows, prev] = -np.eye(2)
        Du[rows, curr] = np.eye(2)
    return Du


# ── QP formulation ─────────────────────────────────────────────────────
class MPCSolver:
    """Pre-builds matrices; call .solve() each step."""

    def __init__(self, A, B, N, w_pos, w_vel, w_u, w_du, amax):
        self.nx = nx = A.shape[0]
        self.nu = nu = B.shape[1]
        self.N = N
        self.amax = amax

        self.Phi, self.Gamma = build_prediction_matrices(A, B, N)
        self.Du = diff_matrix_2d(N)

        # Stage cost weight
        Qk = np.diag([w_pos, w_pos, w_vel, w_vel])
        self.Q = np.kron(np.eye(N + 1), Qk)

        # Input cost
        self.R = w_u * np.eye(nu * N)

        # Input-smoothing cost
        self.w_du = w_du

        # H = Gamma' Q Gamma + R + w_du Du' Du
        GtQ = self.Gamma.T @ self.Q
        self.H_base = GtQ @ self.Gamma + self.R + w_du * (self.Du.T @ self.Du)

        # First-step smoothing block
        E0 = np.zeros((nu, nu * N))
        E0[:, 0:nu] = np.eye(nu)
        self.E0 = E0
        self.H = self.H_base + w_du * (E0.T @ E0)
        self.H_sym = (self.H + self.H.T) / 2.0

        self.GtQ = GtQ  # for linear term

        # OSQP setup (if available)
        self._osqp_solver = None
        if _HAS_OSQP:
            self._setup_osqp()

    def _setup_osqp(self):
        n = self.nu * self.N
        P = sparse.csc_matrix(self.H_sym)
        # Bounds: identity constraint
        A_ineq = sparse.eye(n, format='csc')
        l = -self.amax * np.ones(n)
        u = self.amax * np.ones(n)
        self._osqp_solver = osqp.OSQP()
        q_dummy = np.zeros(n)
        self._osqp_solver.setup(P, q_dummy, A_ineq, l, u,
                                verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                                max_iter=500, warm_start=True)

    def solve(self, xd: np.ndarray, u_prev: np.ndarray,
              pref: np.ndarray, vref: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Solve MPC QP.

        Args:
            xd:    [4] current drone state
            u_prev: [2] previous accel command
            pref:  [2, N+1] reference positions
            vref:  [2, N+1] reference velocities

        Returns:
            (U_optimal or None, solve_time_ms)
        """
        t0 = time.perf_counter()

        nx, nu, N = self.nx, self.nu, self.N

        # Build desired state vector
        xdes = np.zeros(nx * (N + 1))
        for i in range(N + 1):
            idx = slice(i * nx, (i + 1) * nx)
            xdes[idx] = np.array([pref[0, i], pref[1, i],
                                  vref[0, i], vref[1, i]])

        # Linear term: f = Gamma' Q (Phi x - xdes) - w_du E0' u_prev
        residual = self.Phi @ xd - xdes
        f = self.GtQ @ residual - self.w_du * (self.E0.T @ u_prev)

        if _HAS_OSQP and self._osqp_solver is not None:
            self._osqp_solver.update(q=f)
            res = self._osqp_solver.solve()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if res.info.status == 'solved' or res.info.status == 'solved_inaccurate':
                return res.x, dt_ms
            return None, dt_ms
        else:
            # Fallback: scipy trust-constr
            n = nu * N
            bounds = Bounds(-self.amax * np.ones(n), self.amax * np.ones(n))

            def cost_and_grad(U):
                c = 0.5 * U @ self.H_sym @ U + f @ U
                g = self.H_sym @ U + f
                return c, g

            x0 = np.zeros(n)
            res = minimize(cost_and_grad, x0, jac=True, method='trust-constr',
                           bounds=bounds,
                           options={'maxiter': 200, 'gtol': 1e-5, 'verbose': 0})
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if res.success or res.fun < 1e10:
                return res.x, dt_ms
            return None, dt_ms

    def get_predicted_trajectory(self, xd: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Return predicted state trajectory [(N+1), 4]."""
        X = self.Phi @ xd + self.Gamma @ U
        return X.reshape(self.N + 1, self.nx)
