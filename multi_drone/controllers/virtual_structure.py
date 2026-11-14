"""
Virtual Structure (VS) controller.

The formation is treated as a rigid virtual body whose centroid tracks
the target via MPC. Each drone is assigned a node in this virtual body
and uses MPC to track its node's world position.

A fallback potential-field layer handles local collision avoidance when
the virtual node position is obstructed.
"""
import numpy as np
from typing import List, Optional

from controllers import FormationController
from mpc_solver import MPCSolver
from dynamics import make_AB
from formation import (get_formation_offsets, formation_world_positions,
                       compute_formation_heading)
from potential_field import (obstacle_repulsive_force,
                             inter_drone_repulsive_force)


class VirtualStructureController(FormationController):
    name = "VS"

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_drones = cfg.formation.n_drones
        self.offsets = get_formation_offsets(self.n_drones)
        A, B = make_AB(cfg.dt)

        # Each drone gets its own MPC solver
        self.solvers = []
        for _ in range(self.n_drones):
            s = MPCSolver(A, B, cfg.mpc.N,
                          cfg.mpc.w_pos, cfg.mpc.w_vel,
                          cfg.mpc.w_u, cfg.mpc.w_du,
                          cfg.drone.amax)
            self.solvers.append(s)

        self.u_prev = [np.zeros(2) for _ in range(self.n_drones)]
        self._node_positions = None

    def reset(self):
        self.u_prev = [np.zeros(2) for _ in range(self.n_drones)]
        self._node_positions = None

    def compute_controls(self, drone_states, target_state, obstacles, dt, t,
                         **kwargs) -> List[np.ndarray]:
        cfg = self.cfg
        N = cfg.mpc.N
        R_des = cfg.R_des

        # Virtual body centroid = R_des behind target (following distance)
        tgt_pos = target_state[0:2]
        tgt_vel = target_state[2:4]
        heading = compute_formation_heading(tgt_vel)
        tgt_speed = np.linalg.norm(tgt_vel)
        if tgt_speed > 0.1:
            tgt_dir = tgt_vel / tgt_speed
        else:
            tgt_dir = np.array([1.0, 0.0])
        follow_centre = tgt_pos - R_des * tgt_dir

        # Compute world positions for each virtual node
        node_positions = formation_world_positions(follow_centre, heading, self.offsets)
        self._node_positions = node_positions

        accels = []
        for i in range(self.n_drones):
            node = node_positions[i]
            node_vel = tgt_vel.copy()  # nodes move with target velocity

            # Build reference: constant-velocity prediction of the following node
            pref = np.zeros((2, N + 1))
            vref = np.zeros((2, N + 1))
            for j in range(N + 1):
                dt_fwd = j * dt
                pref[:, j] = node + node_vel * dt_fwd
                vref[:, j] = node_vel

            # Solve MPC for this drone
            U, solve_ms = self.solvers[i].solve(
                drone_states[i], self.u_prev[i], pref, vref)

            if U is not None:
                a_des = U[0:2]
            else:
                a_des = np.zeros(2)

            # APF collision avoidance layer
            other_pos = [drone_states[j, 0:2] for j in range(self.n_drones) if j != i]
            f_rep_obs = obstacle_repulsive_force(drone_states[i, 0:2], obstacles)
            f_rep_drone = inter_drone_repulsive_force(drone_states[i, 0:2], other_pos)
            a_apf = f_rep_obs + f_rep_drone

            a_total = a_des + a_apf
            mag = np.linalg.norm(a_total)
            if mag > cfg.drone.amax:
                a_total = a_total * (cfg.drone.amax / mag)

            self.u_prev[i] = a_total
            accels.append(a_total)

        return accels

    def get_formation_node_positions(self) -> Optional[np.ndarray]:
        return self._node_positions
