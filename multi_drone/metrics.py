"""
Metrics computation for a single trial.
"""
import numpy as np
from typing import Dict, Any, Optional


def compute_trial_metrics(log: dict, cfg, duration: float) -> Dict[str, Any]:
    """
    Compute all required metrics from a trial log dictionary.

    log keys expected:
        t:              [K] time array
        drone_states:   [K, N, 4] drone states
        target_states:  [K, 4] target states
        accels:         [K, N, 2] applied accelerations
        solve_times_ms: [K, N] MPC solve times per drone
        node_positions: [K, N, 2] or None — assigned formation positions
        collisions:     [K] bool — any drone in an obstacle
        comm_bytes:     scalar total bytes sent

    Returns dict of metric values.
    """
    K = len(log["t"])
    n_drones = log["drone_states"].shape[1]
    dt = cfg.dt

    # ── Capture time ─────────────────────────────────────────────────
    capture_time = None
    time_to_first_contact = None
    for k in range(K):
        for i in range(n_drones):
            dist = np.linalg.norm(
                log["drone_states"][k, i, 0:2] - log["target_states"][k, 0:2])
            if dist <= cfg.capture_radius:
                if time_to_first_contact is None:
                    time_to_first_contact = log["t"][k]
                capture_time = log["t"][k]
                break
        if capture_time is not None:
            break

    # ── Formation error ──────────────────────────────────────────────
    if log.get("node_positions") is not None and log["node_positions"] is not None:
        nodes = log["node_positions"]  # [K, N, 2]
        form_errors = np.linalg.norm(
            log["drone_states"][:, :, 0:2] - nodes, axis=2)  # [K, N]
        formation_error_mean = float(np.mean(form_errors))
        formation_error_std = float(np.std(form_errors))
    else:
        formation_error_mean = None
        formation_error_std = None

    # ── Tracking RMSE ────────────────────────────────────────────────
    # Best drone RMSE
    best_drone_errors = np.inf * np.ones(K)
    for i in range(n_drones):
        dists = np.linalg.norm(
            log["drone_states"][:, i, 0:2] - log["target_states"][:, 0:2], axis=1)
        best_drone_errors = np.minimum(best_drone_errors, dists)
    tracking_rmse = float(np.sqrt(np.mean(best_drone_errors**2)))

    # Centroid RMSE
    centroids = np.mean(log["drone_states"][:, :, 0:2], axis=1)  # [K, 2]
    centroid_dists = np.linalg.norm(
        centroids - log["target_states"][:, 0:2], axis=1)
    centroid_rmse = float(np.sqrt(np.mean(centroid_dists**2)))

    # ── Overshoot ────────────────────────────────────────────────────
    if log.get("node_positions") is not None and log["node_positions"] is not None:
        form_errors_all = np.linalg.norm(
            log["drone_states"][:, :, 0:2] - log["node_positions"], axis=2)
        overshoot_threshold = 3.0  # m beyond formation offset
        overshoot_mean = float(np.mean(np.maximum(form_errors_all - overshoot_threshold, 0)))
    else:
        overshoot_mean = None

    # ── Control effort ───────────────────────────────────────────────
    accel_mags = np.linalg.norm(log["accels"], axis=2)  # [K, N]
    max_control_effort = float(np.max(accel_mags))

    # ── Control latency ──────────────────────────────────────────────
    solve_times = log["solve_times_ms"]  # [K, N]
    all_times = solve_times.flatten()
    all_times = all_times[all_times > 0]
    if len(all_times) > 0:
        control_latency_mean = float(np.mean(all_times))
        control_latency_95pct = float(np.percentile(all_times, 95))
    else:
        control_latency_mean = 0.0
        control_latency_95pct = 0.0

    # ── Collisions and near misses ───────────────────────────────────
    collisions_count = int(np.sum(log.get("obstacle_collisions", np.zeros(K))))

    # Inter-drone near misses
    near_miss_count = 0
    inter_drone_collisions = 0
    for k in range(K):
        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                dist = np.linalg.norm(
                    log["drone_states"][k, i, 0:2] - log["drone_states"][k, j, 0:2])
                if dist < cfg.near_miss_radius:
                    near_miss_count += 1
                if dist < 0.5:  # collision threshold
                    inter_drone_collisions += 1

    collisions_count += inter_drone_collisions

    # ── Energy proxy ─────────────────────────────────────────────────
    energy_estimate = float(np.sum(accel_mags**2) * dt)

    # ── Communication ────────────────────────────────────────────────
    comm_tx_bytes = log.get("comm_bytes_sent", 0)

    # ── Distance maintenance (follow at R_des) ────────────────────────
    min_dists = log.get("min_dist_to_target")
    if min_dists is not None and len(min_dists) > 0:
        dist_errors = min_dists - cfg.R_des
        dist_maintenance_rmse = float(np.sqrt(np.mean(dist_errors**2)))
        dist_maintenance_mean_abs = float(np.mean(np.abs(dist_errors)))
        fraction_within_tracking = float(np.mean(min_dists <= cfg.R_des * 1.2))
    else:
        dist_maintenance_rmse = None
        dist_maintenance_mean_abs = None
        fraction_within_tracking = None

    return {
        "capture_time": capture_time,
        "time_to_first_contact": time_to_first_contact,
        "formation_error_mean": formation_error_mean,
        "formation_error_std": formation_error_std,
        "tracking_RMSE": tracking_rmse,
        "centroid_RMSE": centroid_rmse,
        "overshoot_mean": overshoot_mean,
        "max_control_effort": max_control_effort,
        "control_latency_mean_ms": control_latency_mean,
        "control_latency_95pct_ms": control_latency_95pct,
        "collisions_count": collisions_count,
        "near_miss_count": near_miss_count,
        "energy_estimate": energy_estimate,
        "comm_tx_bytes": comm_tx_bytes,
        "duration_s": duration,
        "dist_maintenance_rmse": dist_maintenance_rmse,
        "dist_maintenance_mean_abs": dist_maintenance_mean_abs,
        "fraction_within_tracking": fraction_within_tracking,
    }
