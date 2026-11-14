"""
Consensus + APF (distributed) controller.

No fixed leader. All drones run a consensus protocol for shared formation
centroid and a relative offset assignment. Combined with artificial potential
fields for obstacle and target avoidance. Supports degraded comms.
"""
import numpy as np
from typing import List, Optional

from controllers import FormationController
from formation import (get_formation_offsets, formation_world_positions,
                       compute_formation_heading)
from potential_field import combined_apf_accel
from comms import CommChannel


class ConsensusAPFController(FormationController):
    name = "Consensus"

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_drones = cfg.formation.n_drones
        self.offsets = get_formation_offsets(self.n_drones)

        # Each drone's local estimate of the formation centroid
        self.centroid_estimates = None
        self.vel_estimates = None

        # Consensus gain
        self.alpha = 0.3  # consensus weighting on neighbours

        self._node_positions = None

    def reset(self):
        self.centroid_estimates = None
        self.vel_estimates = None
        self._node_positions = None

    def compute_controls(self, drone_states, target_state, obstacles, dt, t,
                         comms: Optional[CommChannel] = None,
                         **kwargs) -> List[np.ndarray]:
        cfg = self.cfg
        n = self.n_drones
        tgt_pos = target_state[0:2]
        tgt_vel = target_state[2:4]
        R_des = cfg.R_des
        tgt_speed = np.linalg.norm(tgt_vel)
        if tgt_speed > 0.1:
            tgt_dir = tgt_vel / tgt_speed
        else:
            tgt_dir = np.array([1.0, 0.0])
        follow_centre = tgt_pos - R_des * tgt_dir

        # Initialize centroid estimates
        if self.centroid_estimates is None:
            self.centroid_estimates = np.tile(follow_centre, (n, 1))
            self.vel_estimates = np.tile(tgt_vel, (n, 1))

        # --- Communication: each drone broadcasts its state and centroid estimate ---
        if comms is not None:
            for i in range(n):
                comms.send(i, {
                    "pos": drone_states[i, 0:2].tolist(),
                    "vel": drone_states[i, 2:4].tolist(),
                    "centroid_est": self.centroid_estimates[i].tolist(),
                    "vel_est": self.vel_estimates[i].tolist(),
                }, t)

        # --- Consensus update ---
        for i in range(n):
            if comms is not None:
                received = comms.receive(i, t)
            else:
                # Ideal comms: receive from all others
                received = {}
                for j in range(n):
                    if j != i:
                        received[j] = {
                            "centroid_est": self.centroid_estimates[j].tolist(),
                            "vel_est": self.vel_estimates[j].tolist(),
                        }

            if received:
                neighbour_centroids = np.array(
                    [received[k]["centroid_est"] for k in received])
                neighbour_vels = np.array(
                    [received[k]["vel_est"] for k in received])
                mean_c = np.mean(neighbour_centroids, axis=0)
                mean_v = np.mean(neighbour_vels, axis=0)
                self.centroid_estimates[i] = (
                    (1 - self.alpha) * self.centroid_estimates[i] +
                    self.alpha * mean_c)
                self.vel_estimates[i] = (
                    (1 - self.alpha) * self.vel_estimates[i] +
                    self.alpha * mean_v)

            # Also pull toward following position (R_des behind target)
            self.centroid_estimates[i] = (
                0.8 * self.centroid_estimates[i] + 0.2 * follow_centre)
            self.vel_estimates[i] = (
                0.8 * self.vel_estimates[i] + 0.2 * tgt_vel)

        # --- Compute assigned positions and APF-based control ---
        heading = compute_formation_heading(tgt_vel)
        accels = []
        node_positions = np.zeros((n, 2))

        for i in range(n):
            # Assigned position = local centroid estimate + rotated offset
            c, s = np.cos(heading), np.sin(heading)
            R = np.array([[c, -s], [s, c]])
            assigned_pos = self.centroid_estimates[i] + R @ self.offsets[i]
            node_positions[i] = assigned_pos

            # APF: attract to assigned position, repel from obstacles and drones
            other_pos = [drone_states[j, 0:2] for j in range(n) if j != i]
            a = combined_apf_accel(
                drone_states[i, 0:2], assigned_pos, other_pos, obstacles,
                cfg.drone.amax,
                k_att=3.0, k_rep_obs=8.0, k_rep_drone=10.0,
                d0_obs=5.0, d_safe_drone=3.0)

            # Add velocity damping toward formation velocity
            vel_err = self.vel_estimates[i] - drone_states[i, 2:4]
            a += 0.5 * vel_err

            mag = np.linalg.norm(a)
            if mag > cfg.drone.amax:
                a = a * (cfg.drone.amax / mag)

            accels.append(a)

        self._node_positions = node_positions
        return accels

    def get_formation_node_positions(self) -> Optional[np.ndarray]:
        return self._node_positions
