"""
Simulation parameters and experiment configurations.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class DroneParams:
    mass: float = 2.5          # kg
    vmax: float = 15.0         # m/s
    amax: float = 5.0          # m/s^2
    blade_dist: float = 0.40   # m
    tilt_max_deg: float = 30.0
    tilt_tau: float = 0.20     # s  (first-order time constant)
    tilt_rate_max_deg: float = 200.0
    g: float = 9.81
    rho: float = 1.225         # air density kg/m^3
    A_disc: float = 0.04       # rotor disc area m^2
    altitude: float = 20.0     # fixed z (m)

    @property
    def tilt_max(self):
        return np.deg2rad(self.tilt_max_deg)

    @property
    def tilt_rate_max(self):
        return np.deg2rad(self.tilt_rate_max_deg)


@dataclass
class MPCParams:
    N: int = 30                # prediction horizon
    w_pos: float = 15.0
    w_vel: float = 0.1
    w_u: float = 2.93
    w_du: float = 3.13


@dataclass
class FormationParams:
    """Triangular formation: leader at front, two followers behind."""
    offsets: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 0.0),       # leader (drone 0): at formation centre front
        (-5.0, -5.0),     # follower left  (drone 1): 5m behind, 5m left
        (-5.0,  5.0),     # follower right (drone 2): 5m behind, 5m right
    ])
    n_drones: int = 3


@dataclass
class CorridorParams:
    length: float = 300.0         # total path length (m)
    width: float = 12.0           # nominal corridor width (m)
    turn_position: float = 150.0  # distance along corridor where turn occurs
    turn_angle_deg: float = 90.0  # right turn
    turn_width: float = 12.0      # width at turn (can be narrowed)
    wall_thickness: float = 3.0   # building wall thickness (m)


@dataclass
class NoiseParams:
    sigma_pos: float = 0.1   # position measurement noise std (m)
    sigma_vel: float = 0.05  # velocity measurement noise std (m/s)
    kf_q_intensity: float = 0.5  # KF process noise intensity


@dataclass
class CommsParams:
    drop_rate: float = 0.0     # packet drop probability [0,1]
    latency_ms: float = 0.0    # communication latency (ms)
    msg_size_bytes: int = 64   # size of one state message (bytes)


@dataclass
class SimConfig:
    dt: float = 0.05           # time step (s)
    t_max: float = 120.0       # max simulation duration (s)
    capture_radius: float = 1.5  # first-contact threshold (m) — metric only, sim continues
    R_des: float = 8.0           # desired following distance behind target (m)
    target_speed: float = 5.0  # target speed (m/s)
    near_miss_radius: float = 2.0  # near-miss threshold (m)
    safety_inflation: float = 1.0
    safety_candidates: int = 24
    emergency_brake: bool = True

    drone: DroneParams = field(default_factory=DroneParams)
    mpc: MPCParams = field(default_factory=MPCParams)
    formation: FormationParams = field(default_factory=FormationParams)
    corridor: CorridorParams = field(default_factory=CorridorParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    comms: CommsParams = field(default_factory=CommsParams)

    # Video
    video_fps: int = 30
    video_res: Tuple[int, int] = (1280, 720)
    save_video: bool = True

    @property
    def n_steps(self):
        return int(self.t_max / self.dt) + 1


# ── Experiment presets ──────────────────────────────────────────────────
def nominal_config() -> SimConfig:
    return SimConfig()


def comms_degraded_config() -> SimConfig:
    cfg = SimConfig()
    cfg.comms = CommsParams(drop_rate=0.3, latency_ms=100.0)
    return cfg


def high_noise_config() -> SimConfig:
    cfg = SimConfig()
    cfg.noise = NoiseParams(sigma_pos=0.5, sigma_vel=0.05)
    return cfg


def narrow_turn_config() -> SimConfig:
    cfg = SimConfig()
    cfg.corridor = CorridorParams(
        length=300.0, width=12.0, turn_position=150.0,
        turn_angle_deg=90.0, turn_width=8.0, wall_thickness=3.0,
    )
    return cfg


def simple_turn_config() -> SimConfig:
    """A shorter corridor (100m) with a 90-degree turn at 50m."""
    cfg = SimConfig()
    cfg.corridor = CorridorParams(
        length=100.0, width=12.0, turn_position=50.0,
        turn_angle_deg=90.0, turn_width=12.0, wall_thickness=3.0,
    )
    return cfg


EXPERIMENT_CONFIGS = {
    "nominal":        nominal_config,
    "comms_degraded": comms_degraded_config,
    "high_noise":     high_noise_config,
    "narrow_turn":    narrow_turn_config,
    "simple_turn":    simple_turn_config,
}

METHOD_NAMES = ["VS", "LF", "Consensus", "Hybrid", "Baseline"]
SEEDS = [1, 2, 3, 4, 5]
