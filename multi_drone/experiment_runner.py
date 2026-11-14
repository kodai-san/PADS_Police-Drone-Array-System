"""
Experiment runner — runs all methods × experiments × seeds and saves outputs.

Usage:
    python experiment_runner.py                          # run all
    python experiment_runner.py --method VS              # single method
    python experiment_runner.py --method VS --experiment nominal --seed 1
    python experiment_runner.py --duration 20            # short test run
"""
import argparse
import json
import csv
import os
import sys
import numpy as np
from pathlib import Path

from config import (SimConfig, EXPERIMENT_CONFIGS, METHOD_NAMES, SEEDS,
                    nominal_config, comms_degraded_config,
                    high_noise_config, narrow_turn_config)
from simulator import run_trial


def _make_controller(method: str, cfg: SimConfig):
    """Instantiate the formation controller for the given method name."""
    if method == "VS":
        from controllers.virtual_structure import VirtualStructureController
        return VirtualStructureController(cfg)
    elif method == "LF":
        from controllers.leader_follower import LeaderFollowerController
        return LeaderFollowerController(cfg)
    elif method == "Consensus":
        from controllers.consensus_apf import ConsensusAPFController
        return ConsensusAPFController(cfg)
    elif method == "Hybrid":
        from controllers.hybrid import HybridController
        return HybridController(cfg)
    elif method == "Baseline":
        from controllers.baseline import BaselineController
        return BaselineController(cfg)
    else:
        raise ValueError(f"Unknown method: {method}")


def _save_csv_log(log: dict, filepath: str):
    """Write timestamped CSV log."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    K = len(log["t"])
    n_drones = log["drone_states"].shape[1]

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        headers = ["time_s"]
        for i in range(n_drones):
            for col in ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
                         "yaw", "controller_latency_ms", "comm_tx_bytes"]:
                headers.append(f"drone{i}_{col}")
        headers += ["target_pos_x", "target_pos_y", "target_pos_z",
                     "target_vel_x", "target_vel_y", "target_vel_z"]
        writer.writerow(headers)

        for k in range(K):
            row = [f"{log['t'][k]:.4f}"]
            for i in range(n_drones):
                s = log["drone_states"][k, i]
                row += [f"{s[0]:.4f}", f"{s[1]:.4f}", "20.0000",  # fixed z
                        f"{s[2]:.4f}", f"{s[3]:.4f}", "0.0000",   # vz=0
                        "0.0000",  # yaw
                        f"{log['solve_times_ms'][k, i]:.3f}",
                        str(log.get("comm_bytes_sent", 0))]
            ts = log["target_states"][k]
            row += [f"{ts[0]:.4f}", f"{ts[1]:.4f}", "0.0000",
                    f"{ts[2]:.4f}", f"{ts[3]:.4f}", "0.0000"]
            writer.writerow(row)


def _save_metrics_json(metrics: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)


def run_single(method: str, experiment: str, seed: int,
               results_dir: str = "results", duration: float = None,
               save_video: bool = True) -> dict:
    """Run a single trial and save all outputs."""
    cfg_fn = EXPERIMENT_CONFIGS[experiment]
    cfg = cfg_fn()
    if duration is not None:
        cfg.t_max = duration

    trial_dir = os.path.join(results_dir, method, experiment, f"seed{seed}")
    os.makedirs(trial_dir, exist_ok=True)

    video_path = os.path.join(trial_dir, f"{method}_{experiment}_seed{seed}.mp4")
    csv_path = os.path.join(trial_dir, f"{method}_{experiment}_seed{seed}_log.csv")
    json_path = os.path.join(trial_dir, f"{method}_{experiment}_seed{seed}_metrics.json")

    print(f"\n{'='*60}")
    print(f"  {method} | {experiment} | seed={seed}")
    print(f"{'='*60}")

    controller = _make_controller(method, cfg)
    result = run_trial(controller, cfg, seed,
                       save_video=save_video, video_path=video_path)

    # Save CSV log
    _save_csv_log(result["log"], csv_path)
    print(f"  CSV log saved: {csv_path}")

    # Save metrics JSON
    _save_metrics_json(result["metrics"], json_path)
    print(f"  Metrics saved: {json_path}")

    cap_str = f"{result['metrics']['capture_time']:.1f}s" if result['metrics']['capture_time'] else "NOT CAPTURED"
    print(f"  Capture: {cap_str}")
    print(f"  Formation error: {result['metrics']['formation_error_mean']}")
    print(f"  Tracking RMSE: {result['metrics']['tracking_RMSE']:.2f} m")
    print(f"  Collisions: {result['metrics']['collisions_count']}")
    print(f"  Control latency: {result['metrics']['control_latency_mean_ms']:.1f} ms "
          f"(95th: {result['metrics']['control_latency_95pct_ms']:.1f} ms)")

    return result


def run_all(results_dir: str = "results", methods=None, experiments=None,
            seeds=None, duration=None, save_video=True):
    """Run the full experiment batch."""
    methods = methods or METHOD_NAMES
    experiments = experiments or list(EXPERIMENT_CONFIGS.keys())
    seeds = seeds or SEEDS

    all_results = []

    total = len(methods) * len(experiments) * len(seeds)
    count = 0

    for method in methods:
        for experiment in experiments:
            for seed in seeds:
                count += 1
                print(f"\n[{count}/{total}]")
                result = run_single(method, experiment, seed,
                                    results_dir, duration, save_video)
                all_results.append({
                    "method": method,
                    "experiment": experiment,
                    "seed": seed,
                    **result["metrics"],
                })

    # Save aggregated CSV
    agg_path = os.path.join(results_dir, "aggregated_results.csv")
    if all_results:
        keys = all_results[0].keys()
        with open(agg_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nAggregated results saved: {agg_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-drone formation experiment runner")
    parser.add_argument("--method", type=str, default=None,
                        help="Single method to run (VS, LF, Consensus, Hybrid, Baseline)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Single experiment (nominal, comms_degraded, high_noise, narrow_turn)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override simulation duration (seconds)")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    args = parser.parse_args()

    methods = [args.method] if args.method else None
    experiments = [args.experiment] if args.experiment else None
    seeds = [args.seed] if args.seed is not None else None

    run_all(args.results_dir, methods, experiments, seeds,
            args.duration, save_video=not args.no_video)


if __name__ == "__main__":
    main()
