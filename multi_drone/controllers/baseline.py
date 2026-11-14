"""
Baseline controller: independent MPC per drone, no formation constraints.

Each drone independently runs MPC to chase the target. No coordination,
no formation maintenance. This is the worst-case baseline for formation
quality but potentially fastest for individual capture.
"""
import numpy as np
from typing import List, Optional

from controllers import FormationController
from mpc_solver import MPCSolver
from dynamics import make_AB


class BaselineController(FormationController):
    name = "Baseline"

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_drones = cfg.formation.n_drones
        A, B = make_AB(cfg.dt)

        self.solvers = []
        for _ in range(self.n_drones):
            s = MPCSolver(A, B, cfg.mpc.N,
                          cfg.mpc.w_pos, cfg.mpc.w_vel,
                          cfg.mpc.w_u, cfg.mpc.w_du,
                          cfg.drone.amax)
            self.solvers.append(s)

        self.u_prev = [np.zeros(2) for _ in range(self.n_drones)]

    def reset(self):
        self.u_prev = [np.zeros(2) for _ in range(self.n_drones)]

    def compute_controls(self, drone_states, target_state, obstacles, dt, t,
                         **kwargs) -> List[np.ndarray]:
        cfg = self.cfg
        N = cfg.mpc.N
        R_des = cfg.R_des
        tgt_pos = target_state[0:2]
        tgt_vel = target_state[2:4]
        tgt_speed = np.linalg.norm(tgt_vel)
        if tgt_speed > 0.1:
            tgt_dir = tgt_vel / tgt_speed
        else:
            tgt_dir = np.array([1.0, 0.0])

        accels = []
        for i in range(self.n_drones):
            # Each drone independently follows R_des behind target
            pref = np.zeros((2, N + 1))
            vref = np.zeros((2, N + 1))
            for j in range(N + 1):
                dt_fwd = j * dt
                future_tgt = tgt_pos + tgt_vel * dt_fwd
                pref[:, j] = future_tgt - R_des * tgt_dir
                vref[:, j] = tgt_vel

            U, solve_ms = self.solvers[i].solve(
                drone_states[i], self.u_prev[i], pref, vref)

            if U is not None:
                a_des = U[0:2]
            else:
                a_des = np.zeros(2)

            # Clamp
            mag = np.linalg.norm(a_des)
            if mag > cfg.drone.amax:
                a_des = a_des * (cfg.drone.amax / mag)

            self.u_prev[i] = a_des
            accels.append(a_des)

        return accels

    def get_formation_node_positions(self) -> Optional[np.ndarray]:
        """No formation nodes — all drones chase independently."""
        return None
