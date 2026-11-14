"""
Leader-Follower (LF) controller.

Leader (drone 0) uses MPC to chase the target.
Followers maintain fixed offsets relative to leader, with reactive
potential-field collision avoidance and safe backup trajectories.
"""
import numpy as np
from typing import List, Optional

from controllers import FormationController
from mpc_solver import MPCSolver
from dynamics import make_AB
from formation import get_formation_offsets, compute_formation_heading
from potential_field import (obstacle_repulsive_force,
                             inter_drone_repulsive_force)


class LeaderFollowerController(FormationController):
    name = "LF"

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_drones = cfg.formation.n_drones
        self.offsets = get_formation_offsets(self.n_drones)
        A, B = make_AB(cfg.dt)

        # MPC for leader and followers
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
        tgt_pos = target_state[0:2]
        tgt_vel = target_state[2:4]
        tgt_speed = np.linalg.norm(tgt_vel)
        if tgt_speed > 0.1:
            tgt_dir = tgt_vel / tgt_speed
        else:
            tgt_dir = np.array([1.0, 0.0])

        accels = []
        leader_pos = drone_states[0, 0:2]
        leader_vel = drone_states[0, 2:4]

        # Heading from leader velocity for offset rotation
        heading = compute_formation_heading(leader_vel)
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s], [s, c]])

        node_positions = np.zeros((self.n_drones, 2))

        for i in range(self.n_drones):
            if i == 0:
                # Leader: MPC tracking R_des behind target (following at distance)
                follow_pos = tgt_pos - R_des * tgt_dir
                pref = np.zeros((2, N + 1))
                vref = np.zeros((2, N + 1))
                for j in range(N + 1):
                    dt_fwd = j * dt
                    future_tgt = tgt_pos + tgt_vel * dt_fwd
                    pref[:, j] = future_tgt - R_des * tgt_dir
                    vref[:, j] = tgt_vel
                node_positions[i] = follow_pos
            else:
                # Follower: track leader_pos + rotated_offset
                offset_world = R @ self.offsets[i]
                follow_pos = leader_pos + offset_world
                follow_vel = leader_vel.copy()

                pref = np.zeros((2, N + 1))
                vref = np.zeros((2, N + 1))
                for j in range(N + 1):
                    dt_fwd = j * dt
                    pref[:, j] = follow_pos + follow_vel * dt_fwd
                    vref[:, j] = follow_vel
                node_positions[i] = follow_pos

            # Solve MPC
            U, solve_ms = self.solvers[i].solve(
                drone_states[i], self.u_prev[i], pref, vref)

            if U is not None:
                a_des = U[0:2]
            else:
                a_des = np.zeros(2)

            # Reactive collision avoidance (potential field)
            other_pos = [drone_states[j, 0:2] for j in range(self.n_drones) if j != i]
            f_rep_obs = obstacle_repulsive_force(drone_states[i, 0:2], obstacles)
            f_rep_drone = inter_drone_repulsive_force(drone_states[i, 0:2], other_pos)

            # Safe backup: if strong repulsion, blend toward braking
            apf_mag = np.linalg.norm(f_rep_obs + f_rep_drone)
            if apf_mag > cfg.drone.amax * 0.5:
                # Engage backup: decelerate toward safe stop
                v = drone_states[i, 2:4]
                nv = np.linalg.norm(v)
                if nv > 0.5:
                    a_brake = -cfg.drone.amax * 0.5 * v / (nv + 1e-12)
                else:
                    a_brake = np.zeros(2)
                blend = min(1.0, apf_mag / (cfg.drone.amax * 2.0))
                a_des = (1 - blend) * a_des + blend * (a_brake + f_rep_obs + f_rep_drone)

            a_total = a_des + f_rep_obs + f_rep_drone
            mag = np.linalg.norm(a_total)
            if mag > cfg.drone.amax:
                a_total = a_total * (cfg.drone.amax / mag)

            self.u_prev[i] = a_total
            accels.append(a_total)

        self._node_positions = node_positions
        return accels

    def get_formation_node_positions(self) -> Optional[np.ndarray]:
        return self._node_positions
