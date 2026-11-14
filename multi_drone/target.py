"""
Target waypoint follower — port of targetStepWaypoint.m
"""
import numpy as np
from typing import Tuple


def target_step_waypoint(state: np.ndarray, wp_idx: int,
                         waypoints: np.ndarray, speed: float,
                         dt: float) -> Tuple[np.ndarray, int]:
    """
    Advance target one time step along waypoints at constant speed.

    Args:
        state:     [px, py, vx, vy]
        wp_idx:    index of next waypoint to head toward
        waypoints: [P, 2] ordered path points
        speed:     target speed (m/s)
        dt:        time step (s)

    Returns:
        (next_state [4], updated wp_idx)
    """
    n_pts = len(waypoints)
    pos = state[0:2].copy()

    if wp_idx >= n_pts:
        return np.array([pos[0], pos[1], 0.0, 0.0]), wp_idx

    goal = waypoints[wp_idx].copy()
    to_goal = goal - pos
    dist_to_goal = np.linalg.norm(to_goal)
    step_dist = speed * dt

    while dist_to_goal < step_dist and wp_idx < n_pts - 1:
        step_dist -= dist_to_goal
        pos = goal.copy()
        wp_idx += 1
        if wp_idx >= n_pts:
            break
        goal = waypoints[wp_idx].copy()
        to_goal = goal - pos
        dist_to_goal = np.linalg.norm(to_goal)

    if wp_idx < n_pts and dist_to_goal > 1e-6:
        direction = to_goal / dist_to_goal
        pos = pos + direction * step_dist
        vel = direction * speed
    else:
        vel = np.zeros(2)

    return np.array([pos[0], pos[1], vel[0], vel[1]]), wp_idx
