"""
Main simulation engine — runs one trial of multi-drone formation chase.
"""
import time
import numpy as np
from typing import Dict, Any, Optional

from config import SimConfig
from dynamics import make_AB, propagate_drone, safety_filter_accel, tilt_step, compute_power
from target import target_step_waypoint
from environment import (create_corridor, point_in_any_obstacle,
                         inflate_obstacles)
from noise import add_position_noise, add_velocity_noise, KalmanFilter2D
from comms import CommChannel
from formation import compute_formation_error
from metrics import compute_trial_metrics
from controllers import FormationController


def run_trial(controller: FormationController,
              cfg: SimConfig,
              seed: int,
              save_video: bool = True,
              video_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run one full simulation trial.

    Returns dict with keys: log, metrics, video_path
    """
    rng = np.random.default_rng(seed)
    n = cfg.formation.n_drones
    dt = cfg.dt
    K = cfg.n_steps

    # ── Environment ──────────────────────────────────────────────────
    env = create_corridor(
        length=cfg.corridor.length,
        width=cfg.corridor.width,
        turn_pos=cfg.corridor.turn_position,
        turn_width=cfg.corridor.turn_width,
        wall_thick=cfg.corridor.wall_thickness)

    obstacles = env["obstacles"]
    obstacles_inflated = inflate_obstacles(obstacles, cfg.safety_inflation)
    waypoints = env["waypoints"]

    # ── Dynamics ─────────────────────────────────────────────────────
    A, B = make_AB(dt)
    controller.reset()

    # ── Initialize target ────────────────────────────────────────────
    tgt_pos = waypoints[0].copy()
    tgt_dir = waypoints[1] - waypoints[0]
    tgt_dir = tgt_dir / (np.linalg.norm(tgt_dir) + 1e-6)
    target_state = np.array([tgt_pos[0], tgt_pos[1],
                             cfg.target_speed * tgt_dir[0],
                             cfg.target_speed * tgt_dir[1]])
    wp_idx = 1

    # ── Initialize drones in formation ───────────────────────────────
    from formation import get_formation_offsets, formation_world_positions, compute_formation_heading
    offsets = get_formation_offsets(n)
    heading = compute_formation_heading(target_state[2:4])
    init_positions = formation_world_positions(
        tgt_pos - 15.0 * tgt_dir,  # formation starts 15m behind target
        heading, offsets)

    drone_states = np.zeros((n, 4))
    for i in range(n):
        drone_states[i, 0:2] = init_positions[i]
        drone_states[i, 2:4] = cfg.target_speed * tgt_dir * 0.5  # start with some velocity

    # Tilt state per drone
    phi = np.zeros(n)
    theta = np.zeros(n)
    u_prev = np.zeros((n, 2))

    # ── Kalman filters ───────────────────────────────────────────────
    drone_kfs = []
    target_kf = KalmanFilter2D(dt, cfg.noise.kf_q_intensity, cfg.noise.sigma_pos)
    target_kf.init_state(target_state)
    for i in range(n):
        kf = KalmanFilter2D(dt, cfg.noise.kf_q_intensity, cfg.noise.sigma_pos)
        kf.init_state(drone_states[i])
        drone_kfs.append(kf)

    # ── Communication channel ────────────────────────────────────────
    comms = CommChannel(
        drop_rate=cfg.comms.drop_rate,
        latency_ms=cfg.comms.latency_ms,
        dt=dt, rng=rng)

    # ── Logging arrays ───────────────────────────────────────────────
    log = {
        "t": np.zeros(K),
        "drone_states": np.zeros((K, n, 4)),
        "target_states": np.zeros((K, 4)),
        "accels": np.zeros((K, n, 2)),
        "solve_times_ms": np.zeros((K, n)),
        "node_positions": np.zeros((K, n, 2)),
        "obstacle_collisions": np.zeros(K, dtype=bool),
        "min_dist_to_target": np.zeros(K),
        "comm_bytes_sent": 0,
    }
    has_nodes = True

    # ── Video recorder ───────────────────────────────────────────────
    recorder = None
    if save_video and video_path:
        from video_recorder import VideoRecorder
        recorder = VideoRecorder(video_path, env, fps=cfg.video_fps)

    # Video: record every N-th step to hit ~30 fps
    steps_per_frame = max(1, int(1.0 / (cfg.video_fps * dt)))

    # ── Main simulation loop ─────────────────────────────────────────
    actual_steps = 0
    captured = False
    first_contact_time = None
    t_current = 0.0

    for k in range(K):
        t_current = k * dt
        log["t"][k] = t_current
        log["drone_states"][k] = drone_states.copy()
        log["target_states"][k] = target_state.copy()

        # Log closest drone distance (no early stop — maintain R_des)
        dists_to_target = [np.linalg.norm(drone_states[i, 0:2] - target_state[0:2])
                           for i in range(n)]
        min_dist = min(dists_to_target)
        log["min_dist_to_target"][k] = min_dist
        if min_dist <= cfg.capture_radius and first_contact_time is None:
            first_contact_time = t_current
            captured = True

        # ── Noisy measurements ───────────────────────────────────────
        drone_states_noisy = drone_states.copy()
        for i in range(n):
            drone_states_noisy[i, 0:2] = add_position_noise(
                drone_states[i, 0:2], cfg.noise.sigma_pos, rng)
            drone_states_noisy[i, 2:4] = add_velocity_noise(
                drone_states[i, 2:4], cfg.noise.sigma_vel, rng)
            # KF predict + update
            drone_kfs[i].predict(u_prev[i], B)
            drone_kfs[i].update(drone_states_noisy[i, 0:2])

        target_noisy_pos = add_position_noise(
            target_state[0:2], cfg.noise.sigma_pos, rng)
        target_kf.predict()
        target_kf.update(target_noisy_pos)

        # Use filtered states for controller
        drone_states_filtered = np.zeros_like(drone_states)
        for i in range(n):
            drone_states_filtered[i] = drone_kfs[i].state
        target_state_filtered = target_kf.state

        # ── Controller ───────────────────────────────────────────────
        t0 = time.perf_counter()
        accels = controller.compute_controls(
            drone_states_filtered, target_state_filtered,
            obstacles, dt, t_current, comms=comms)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        for i in range(n):
            log["solve_times_ms"][k, i] = solve_ms / n  # approximate per-drone

        # ── Apply tilt dynamics and safety filter ────────────────────
        for i in range(n):
            a_des = accels[i]

            # Tilt dynamics
            theta_cmd = np.clip(np.arctan2(a_des[0], cfg.drone.g),
                                -cfg.drone.tilt_max, cfg.drone.tilt_max)
            phi_cmd = np.clip(-np.arctan2(a_des[1], cfg.drone.g),
                              -cfg.drone.tilt_max, cfg.drone.tilt_max)
            phi[i], theta[i] = tilt_step(
                phi[i], theta[i], phi_cmd, theta_cmd,
                cfg.drone.tilt_tau, cfg.drone.tilt_rate_max,
                cfg.drone.tilt_max, dt)

            accel_x = cfg.drone.g * np.tan(theta[i])
            accel_y = -cfg.drone.g * np.tan(phi[i])
            a_act = np.array([accel_x, accel_y])
            a_mag = np.linalg.norm(a_act)
            if a_mag > cfg.drone.amax:
                a_act = a_act * (cfg.drone.amax / a_mag)

            # Safety filter
            a_safe, _ = safety_filter_accel(
                drone_states[i], a_act, A, B,
                obstacles_inflated, cfg.drone.amax,
                n_candidates=cfg.safety_candidates,
                emergency_brake=cfg.emergency_brake,
                point_in_obstacle_fn=point_in_any_obstacle)

            # Propagate
            drone_states[i] = propagate_drone(
                drone_states[i], a_safe, A, B, cfg.drone.vmax)

            # Check obstacle collision (log it but don't stop)
            if point_in_any_obstacle(drone_states[i, 0:2], obstacles):
                log["obstacle_collisions"][k] = True
                drone_states[i, 2:4] = np.zeros(2)  # stop if inside obstacle

            log["accels"][k, i] = a_safe
            u_prev[i] = a_safe

        # ── Formation node positions (for logging) ───────────────────
        node_pos = controller.get_formation_node_positions()
        if node_pos is not None:
            log["node_positions"][k] = node_pos
        else:
            has_nodes = False

        # ── Propagate target ─────────────────────────────────────────
        target_state, wp_idx = target_step_waypoint(
            target_state, wp_idx, waypoints, cfg.target_speed, dt)

        # Stop if target reached end
        if wp_idx >= len(waypoints) and np.linalg.norm(target_state[2:4]) < 0.01:
            actual_steps = k + 1
            break

        # ── Video frame ──────────────────────────────────────────────
        if recorder and k % steps_per_frame == 0:
            dist_status = f"dist={min_dist:.1f}m"
            if min_dist <= cfg.R_des * 1.2:
                dist_status += " ✓TRACKING"
            recorder.add_frame(
                t_current, drone_states[:, 0:2],
                target_state[0:2], node_pos,
                r_des=cfg.R_des,
                extra_text=dist_status)

        actual_steps = k + 1

    # ── Truncate logs ────────────────────────────────────────────────
    for key in ["t", "drone_states", "target_states", "accels",
                "solve_times_ms", "node_positions", "obstacle_collisions",
                "min_dist_to_target"]:
        log[key] = log[key][:actual_steps]

    if not has_nodes:
        log["node_positions"] = None

    log["comm_bytes_sent"] = comms.bytes_sent

    # ── Compute metrics ──────────────────────────────────────────────
    duration = actual_steps * dt
    trial_metrics = compute_trial_metrics(log, cfg, duration)

    # ── Save video ───────────────────────────────────────────────────
    if recorder:
        try:
            recorder.save()
        except Exception as e:
            print(f"  Warning: video save failed: {e}")
        finally:
            recorder.close()

    return {
        "log": log,
        "metrics": trial_metrics,
        "video_path": video_path,
        "captured": captured,
        "first_contact_time": first_contact_time,
        "duration": duration,
    }
