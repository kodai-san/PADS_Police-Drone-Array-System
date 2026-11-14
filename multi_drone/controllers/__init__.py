"""
Base class for formation controllers.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional


class FormationController(ABC):
    """
    Interface for multi-drone formation controllers.

    Each controller receives the full (possibly noisy) state of all drones
    and the target, and returns a 2-D acceleration command per drone.
    """

    name: str = "base"

    @abstractmethod
    def compute_controls(self,
                         drone_states: np.ndarray,
                         target_state: np.ndarray,
                         obstacles: list,
                         dt: float,
                         t: float,
                         **kwargs) -> List[np.ndarray]:
        """
        Args:
            drone_states: [N, 4] array of [px, py, vx, vy] per drone
            target_state: [4] target state [px, py, vx, vy]
            obstacles:    list of polygon arrays
            dt:           time step
            t:            current time

        Returns:
            List of N 2-D acceleration arrays, one per drone.
        """
        ...

    def reset(self):
        """Reset any internal state (called before each trial)."""
        pass

    def get_formation_node_positions(self) -> Optional[np.ndarray]:
        """Return the current assigned formation node positions [N,2], if applicable."""
        return None
