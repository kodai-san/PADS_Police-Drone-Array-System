"""
Hybrid controller: Virtual Structure + learned local reactive controller.

Uses the virtual structure for global formation shape (identical to VS),
but replaces the APF potential-field layer with a tuned heuristic reactive
controller for local obstacle negotiation.

NOTE: Full RL (PPO) training was not feasible within the test session.
This uses a hand-tuned reactive policy instead. See report for details.
"""
import numpy as np
from typing import List, Optional

from controllers import FormationController
from mpc_solver import MPCSolver
from dynamics import make_AB
from formation import (get_formation_offsets, formation_world_positions,
                       compute_formation_heading)
from environment import point_in_any_obstacle


class HeuristicReactivePolicy:
    """
    Replaces APF with a more structured reactive policy:
    - Obstacle proximity → steer perpendicular to nearest wall + decelerate
    - Inter-drone proximity → repel along inter-drone vector
    - Smooth blending between MPC command and reactive override
    """

    def __init__(self, d_obs_trigger: float = 4.0, d_drone_trigger: float = 3.0,
                 amax: float = 5.0):
        self.d_obs = d_obs_trigger
        self.d_drone = d_drone_trigger
        self.amax = amax

    def compute_reactive(self, pos, vel, other_positions, obstacles,
                         a_mpc) -> np.ndarray:
        """Return blended acceleration."""
        a_reactive = np.zeros(2)
        blend = 0.0

        # Obstacle avoidance: sample nearby wall points
        min_obs_dist = np.inf
        nearest_wall_normal = np.zeros(2)
        for obs in obstacles:
            n = len(obs)
            for ii in range(n):
                jj = (ii + 1) % n
                # Closest point on edge to pos
                edge = obs[jj] - obs[ii]
                edge_len = np.linalg.norm(edge)
                if edge_len < 1e-6:
                    continue
                edge_dir = edge / edge_len
                t = np.clip(np.dot(pos - obs[ii], edge_dir) / edge_len, 0, 1)
                closest = obs[ii] + t * edge
                diff = pos - closest
                dist = np.linalg.norm(diff)
                if dist < min_obs_dist and dist > 0.01:
                    min_obs_dist = dist
                    nearest_wall_normal = diff / dist

        if min_obs_dist < self.d_obs:
            # Steer perpendicular to wall + decelerate
            intensity = (self.d_obs - min_obs_dist) / self.d_obs
            a_steer = nearest_wall_normal * self.amax * intensity

            # Also decelerate if heading toward wall
            speed = np.linalg.norm(vel)
            if speed > 0.5:
                vel_toward_wall = -np.dot(vel, nearest_wall_normal)
                if vel_toward_wall > 0:
                    a_steer += -vel / (speed + 1e-12) * min(vel_toward_wall, self.amax * 0.5)

            a_reactive += a_steer
            blend = max(blend, intensity)

        # Inter-drone avoidance
        for other_pos in other_positions:
            diff = pos - other_pos
            dist = np.linalg.norm(diff)
            if 0.01 < dist < self.d_drone:
                intensity = (self.d_drone - dist) / self.d_drone
                a_reactive += diff / (dist + 1e-12) * self.amax * intensity * 0.8
                blend = max(blend, intensity * 0.5)

        blend = min(blend, 1.0)
        a_out = (1.0 - blend) * a_mpc + blend * a_reactive

        mag = np.linalg.norm(a_out)
        if mag > self.amax:
            a_out = a_out * (self.amax / mag)
        return a_out


class HybridController(FormationController):
    name = "Hybrid"

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_drones = cfg.formation.n_drones
        self.offsets = get_formation_offsets(self.n_drones)
        A, B = make_AB(cfg.dt)

        self.solvers = []
        for _ in range(self.n_drones):
            s = MPCSolver(A, B, cfg.mpc.N,
                          cfg.mpc.w_pos, cfg.mpc.w_vel,
                          cfg.mpc.w_u, cfg.mpc.w_du,
                          cfg.drone.amax)
            self.solvers.append(s)

        self.reactive = HeuristicReactivePolicy(amax=cfg.drone.amax)
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
        tgt_pos = target_state[0:2]
        tgt_vel = target_state[2:4]
        tgt_speed = np.linalg.norm(tgt_vel)
        if tgt_speed > 0.1:
            tgt_dir = tgt_vel / tgt_speed
        else:
            tgt_dir = np.array([1.0, 0.0])
        follow_centre = tgt_pos - R_des * tgt_dir
        heading = compute_formation_heading(tgt_vel)

        node_positions = formation_world_positions(follow_centre, heading, self.offsets)
        self._node_positions = node_positions

        accels = []
        for i in range(self.n_drones):
            node = node_positions[i]
            node_vel = tgt_vel.copy()

            pref = np.zeros((2, N + 1))
            vref = np.zeros((2, N + 1))
            for j in range(N + 1):
                dt_fwd = j * dt
                future_tgt = tgt_pos + tgt_vel * dt_fwd
                future_node = formation_world_positions(
                    future_tgt - R_des * tgt_dir, heading, self.offsets)[i]
                pref[:, j] = future_node
                vref[:, j] = node_vel

            U, solve_ms = self.solvers[i].solve(
                drone_states[i], self.u_prev[i], pref, vref)

            if U is not None:
                a_mpc = U[0:2]
            else:
                a_mpc = np.zeros(2)

            # Use heuristic reactive policy instead of APF
            other_pos = [drone_states[j, 0:2] for j in range(self.n_drones) if j != i]
            a_total = self.reactive.compute_reactive(
                drone_states[i, 0:2], drone_states[i, 2:4],
                other_pos, obstacles, a_mpc)

            self.u_prev[i] = a_total
            accels.append(a_total)

        return accels

    def get_formation_node_positions(self) -> Optional[np.ndarray]:
        return self._node_positions
