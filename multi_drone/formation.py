"""
Formation geometry utilities.
"""
import numpy as np
from typing import List, Tuple


def get_formation_offsets(n_drones: int = 3) -> np.ndarray:
    """
    Return Nx2 array of formation offsets in body frame.
    Leader (0) at front, two followers behind.
    Offsets are relative to formation centroid, in the frame where
    +x = forward (direction of motion).
    """
    offsets = np.array([
        [ 0.0,  0.0],   # leader: at the front
        [-2.5, -2.5],   # follower left: 2.5 m behind, 2.5 m left
        [-2.5,  2.5],   # follower right: 2.5 m behind, 2.5 m right
    ])
    return offsets[:n_drones]


def formation_world_positions(centre: np.ndarray, heading: float,
                               offsets: np.ndarray) -> np.ndarray:
    """
    Compute world positions of formation nodes given the formation
    centroid and heading direction.

    Args:
        centre:  [2] world position of formation centroid
        heading: heading angle in radians (0 = +x)
        offsets: [N, 2] body-frame offsets

    Returns:
        [N, 2] world positions of each formation node.
    """
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    world_offsets = offsets @ R.T
    return centre[np.newaxis, :] + world_offsets


def compute_formation_heading(vel: np.ndarray) -> float:
    """Formation heading from velocity vector."""
    speed = np.linalg.norm(vel)
    if speed < 0.1:
        return 0.0
    return np.arctan2(vel[1], vel[0])


def compute_formation_error(drone_positions: np.ndarray,
                            node_positions: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean and std of formation error = distance each drone
    is from its assigned formation node.
    """
    errors = np.linalg.norm(drone_positions - node_positions, axis=1)
    return float(np.mean(errors)), float(np.std(errors))
