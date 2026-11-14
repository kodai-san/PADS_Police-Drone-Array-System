"""
Measurement noise and Kalman filter for state estimation.
"""
import numpy as np
from typing import Optional


def add_position_noise(pos: np.ndarray, sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    return pos + sigma * rng.standard_normal(pos.shape)


def add_velocity_noise(vel: np.ndarray, sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    return vel + sigma * rng.standard_normal(vel.shape)


class KalmanFilter2D:
    """
    Constant-velocity Kalman filter for 2-D position tracking.
    State: [px, py, vx, vy]
    Measurement: [px, py]  (position only)
    """

    def __init__(self, dt: float, q_intensity: float, r_pos: float):
        self.nx = 4
        self.dt = dt

        # State transition
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        # Measurement matrix (position only)
        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # Process noise
        self.Q = q_intensity * np.array([
            [dt**3/3, 0,      dt**2/2, 0      ],
            [0,       dt**3/3, 0,      dt**2/2],
            [dt**2/2, 0,      dt,      0      ],
            [0,       dt**2/2, 0,      dt     ],
        ])

        # Measurement noise
        self.R = (r_pos**2 + 1e-12) * np.eye(2)

        # State and covariance
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1.0

    def init_state(self, x0: np.ndarray):
        self.x = x0.copy()
        self.P = np.eye(4) * 0.1

    def predict(self, u: Optional[np.ndarray] = None,
                B: Optional[np.ndarray] = None):
        """Predict step. Optionally include control input."""
        self.x = self.F @ self.x
        if u is not None and B is not None:
            self.x += B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z_pos: np.ndarray):
        """Update step with position measurement [px, py]."""
        y = z_pos - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def state(self) -> np.ndarray:
        return self.x.copy()
