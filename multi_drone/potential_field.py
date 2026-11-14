"""
Artificial Potential Field (APF) for obstacle and inter-drone avoidance.
Also used as the reactive local layer in VS, LF, and Consensus controllers.
"""
import numpy as np
from typing import List


def obstacle_repulsive_force(pos: np.ndarray, obstacles: list,
                             d0: float = 5.0, k_rep: float = 8.0,
                             n_samples: int = 8) -> np.ndarray:
    """
    Compute repulsive force from nearby obstacle walls.

    Samples points on each obstacle polygon edge and computes repulsion
    if distance < d0.
    """
    force = np.zeros(2)
    for obs in obstacles:
        n = len(obs)
        for i in range(n):
            j = (i + 1) % n
            # Sample along edge
            for t in np.linspace(0, 1, n_samples):
                wall_pt = obs[i] * (1 - t) + obs[j] * t
                diff = pos - wall_pt
                dist = np.linalg.norm(diff)
                if 0.1 < dist < d0:
                    magnitude = k_rep * (1.0 / dist - 1.0 / d0) / (dist**2)
                    force += magnitude * diff / dist
    return force


def inter_drone_repulsive_force(pos: np.ndarray, other_positions: List[np.ndarray],
                                 d_safe: float = 3.0,
                                 k_rep: float = 10.0) -> np.ndarray:
    """Repulsive force between drones. Strong when within d_safe."""
    force = np.zeros(2)
    for other_pos in other_positions:
        diff = pos - other_pos
        dist = np.linalg.norm(diff)
        if 0.01 < dist < d_safe:
            magnitude = k_rep * (1.0 / dist - 1.0 / d_safe) / (dist**2)
            force += magnitude * diff / (dist + 1e-12)
    return force


def target_attractive_force(pos: np.ndarray, target_pos: np.ndarray,
                            k_att: float = 1.0) -> np.ndarray:
    """Simple linear attractive force toward target position."""
    diff = target_pos - pos
    dist = np.linalg.norm(diff)
    if dist < 1e-6:
        return np.zeros(2)
    return k_att * diff


def combined_apf_accel(pos: np.ndarray, target_pos: np.ndarray,
                       other_positions: List[np.ndarray],
                       obstacles: list, amax: float,
                       k_att: float = 2.0, k_rep_obs: float = 8.0,
                       k_rep_drone: float = 10.0,
                       d0_obs: float = 5.0, d_safe_drone: float = 3.0
                       ) -> np.ndarray:
    """Compute APF-based acceleration (clamped to amax)."""
    f_att = target_attractive_force(pos, target_pos, k_att)
    f_rep_obs = obstacle_repulsive_force(pos, obstacles, d0_obs, k_rep_obs)
    f_rep_drone = inter_drone_repulsive_force(pos, other_positions,
                                               d_safe_drone, k_rep_drone)
    total = f_att + f_rep_obs + f_rep_drone
    mag = np.linalg.norm(total)
    if mag > amax:
        total = total * (amax / mag)
    return total
