"""
Drone dynamics: double-integrator in 2-D with tilt and safety filter.

State vector (per drone): [px, py, vx, vy]   (4-element, 2-D planar)
Control input:            [ax, ay]            (2-element acceleration)

For logging we augment with pz (fixed altitude), vz=0, yaw.
"""
import numpy as np
from typing import List, Tuple, Optional


# ── Discrete-time matrices ──────────────────────────────────────────────
def make_AB(dt: float):
    """Return (A, B) for the 2-D double-integrator."""
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=np.float64)
    B = np.array([
        [0.5 * dt**2, 0],
        [0, 0.5 * dt**2],
        [dt, 0],
        [0, dt],
    ], dtype=np.float64)
    return A, B


# ── Tilt dynamics ───────────────────────────────────────────────────────
def _clamp_scalar(x: float, lim: float) -> float:
    return max(-lim, min(lim, x))


def tilt_step(phi: float, theta: float,
              phi_cmd: float, theta_cmd: float,
              tilt_tau: float, tilt_rate_max: float,
              tilt_max: float, dt: float) -> Tuple[float, float]:
    """First-order tilt response with rate and magnitude clamp."""
    phi_dot = (phi_cmd - phi) / tilt_tau
    theta_dot = (theta_cmd - theta) / tilt_tau
    phi_dot = _clamp_scalar(phi_dot, tilt_rate_max)
    theta_dot = _clamp_scalar(theta_dot, tilt_rate_max)
    phi = phi + dt * phi_dot
    theta = theta + dt * theta_dot
    phi = _clamp_scalar(phi, tilt_max)
    theta = _clamp_scalar(theta, tilt_max)
    return phi, theta


# ── Propagation ─────────────────────────────────────────────────────────
def clamp_vec2(v: np.ndarray, vmax: float) -> np.ndarray:
    """Clamp 2-D vector magnitude."""
    n = np.linalg.norm(v)
    if n > vmax:
        return v * (vmax / (n + 1e-12))
    return v


def propagate_drone(state: np.ndarray, accel: np.ndarray,
                    A: np.ndarray, B: np.ndarray,
                    vmax: float) -> np.ndarray:
    """Advance one step: x_{k+1} = A x_k + B u_k, then clamp velocity."""
    x_next = A @ state + B @ accel
    x_next[2:4] = clamp_vec2(x_next[2:4], vmax)
    return x_next


# ── Safety filter (port of safetyFilterAccel.m) ─────────────────────────
def safety_filter_accel(state: np.ndarray, a_cmd: np.ndarray,
                        A: np.ndarray, B: np.ndarray,
                        obstacles: list, amax: float,
                        n_candidates: int = 24,
                        emergency_brake: bool = True,
                        point_in_obstacle_fn=None) -> Tuple[np.ndarray, bool]:
    """
    Find acceleration close to a_cmd that avoids placing the drone
    inside an obstacle at the next time step.

    Returns (a_safe, filter_used).
    """
    if point_in_obstacle_fn is None:
        return a_cmd, False

    # Check if commanded acceleration is safe
    x1 = A @ state + B @ a_cmd
    if not point_in_obstacle_fn(x1[0:2], obstacles):
        return a_cmd, False

    # Build candidates
    candidates = []

    # Braking
    if emergency_brake:
        v = state[2:4]
        nv = np.linalg.norm(v)
        if nv > 1e-3:
            candidates.append(-amax * v / (nv + 1e-12))
        else:
            candidates.append(np.zeros(2))

    # Rotated versions of command
    a0 = a_cmd.copy()
    if np.linalg.norm(a0) < 1e-6:
        a0 = np.array([amax, 0.0])
    a0 = a0 / (np.linalg.norm(a0) + 1e-12) * amax
    for i in range(n_candidates):
        ang = 2 * np.pi * i / n_candidates
        R = np.array([[np.cos(ang), -np.sin(ang)],
                       [np.sin(ang),  np.cos(ang)]])
        candidates.append(R @ a0)

    # Lateral candidates
    v = state[2:4]
    nv = np.linalg.norm(v)
    if nv > 1e-3:
        vhat = v / (nv + 1e-12)
        candidates.append(np.array([-vhat[1], vhat[0]]) * amax)
        candidates.append(np.array([vhat[1], -vhat[0]]) * amax)

    # Pick best safe candidate
    best_a = None
    best_cost = np.inf
    for a in candidates:
        x1 = A @ state + B @ a
        if point_in_obstacle_fn(x1[0:2], obstacles):
            continue
        cost = np.linalg.norm(a - a_cmd)
        if cost < best_cost:
            best_cost = cost
            best_a = a.copy()

    if best_a is None:
        return np.zeros(2), True
    return best_a, True


# ── Energy computation ──────────────────────────────────────────────────
def compute_power(accel: np.ndarray, speed: float,
                  mass: float, g: float,
                  rho: float, A_disc: float) -> float:
    """Compute instantaneous mechanical power (W)."""
    thrust = mass * np.sqrt(accel[0]**2 + accel[1]**2 + g**2)
    P_hover = thrust**1.5 / np.sqrt(2.0 * rho * A_disc)
    P_lateral = np.linalg.norm(accel) * mass * speed
    return P_hover + P_lateral
