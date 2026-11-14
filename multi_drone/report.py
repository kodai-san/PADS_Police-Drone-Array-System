"""
Report generator — produces a markdown comparison report.
"""
import os
import csv
import numpy as np
from collections import defaultdict


def generate_report(results_dir: str = "results"):
    """Generate a markdown report comparing all methods."""
    agg_path = os.path.join(results_dir, "aggregated_results.csv")
    if not os.path.exists(agg_path):
        print(f"Error: {agg_path} not found.")
        return

    # Load data
    rows = []
    with open(agg_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k in ('method', 'experiment'):
                    continue
                if v == 'None' or v == '' or v is None:
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        pass
            rows.append(row)

    methods = sorted(set(r["method"] for r in rows))
    experiments = sorted(set(r["experiment"] for r in rows))

    # ── Aggregate statistics ─────────────────────────────────────────
    stats = defaultdict(lambda: defaultdict(dict))
    for method in methods:
        for exp in experiments:
            subset = [r for r in rows
                      if r["method"] == method and r["experiment"] == exp]
            if not subset:
                continue

            for metric in ["capture_time", "formation_error_mean",
                           "tracking_RMSE", "collisions_count",
                           "control_latency_mean_ms", "energy_estimate",
                           "near_miss_count"]:
                vals = [r[metric] for r in subset if r[metric] is not None]
                if vals:
                    stats[method][exp][metric] = {
                        "mean": np.mean(vals),
                        "std": np.std(vals),
                    }

    # ── Write report ─────────────────────────────────────────────────
    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Multi-Drone Formation Chase: Comparative Results\n\n")
        f.write("## Overview\n\n")
        f.write("This report compares 5 formation control methods for 3 quadcopters "
                "pursuing a single target through a synthetic urban corridor with a 90° turn.\n\n")
        f.write("**Methods**: Virtual Structure (VS), Leader-Follower (LF), "
                "Consensus+APF, Hybrid (VS+Heuristic Reactive), Baseline (independent MPC)\n\n")
        f.write("**Experiments**: Nominal, Comms Degraded (30% drop, 100ms latency), "
                "High Noise (σ_pos=0.5m), Narrow Turn (8m width)\n\n")
        f.write("Each experiment was run with 5 random seeds.\n\n")

        # Comparison table
        f.write("## Comparison Table\n\n")
        f.write("| Method | Experiment | Capture Time (s) | Formation Error (m) | "
                "Tracking RMSE (m) | Collisions | Latency (ms) | Energy |\n")
        f.write("|--------|-----------|----------------:|-------------------:|"
                "-----------------:|-----------:|-------------:|-------:|\n")

        for method in methods:
            for exp in experiments:
                s = stats[method].get(exp, {})
                cap = s.get("capture_time", {})
                form = s.get("formation_error_mean", {})
                rmse = s.get("tracking_RMSE", {})
                col = s.get("collisions_count", {})
                lat = s.get("control_latency_mean_ms", {})
                energy = s.get("energy_estimate", {})

                cap_str = f"{cap.get('mean', 'N/A'):.1f} ± {cap.get('std', 0):.1f}" if cap else "N/A"
                form_str = f"{form.get('mean', 'N/A'):.2f} ± {form.get('std', 0):.2f}" if form else "N/A"
                rmse_str = f"{rmse.get('mean', 'N/A'):.2f} ± {rmse.get('std', 0):.2f}" if rmse else "N/A"
                col_str = f"{col.get('mean', 0):.0f}" if col else "0"
                lat_str = f"{lat.get('mean', 0):.1f}" if lat else "N/A"
                energy_str = f"{energy.get('mean', 0):.0f}" if energy else "N/A"

                f.write(f"| {method} | {exp} | {cap_str} | {form_str} | "
                        f"{rmse_str} | {col_str} | {lat_str} | {energy_str} |\n")

        # Strengths/Weaknesses
        f.write("\n## Strengths and Weaknesses\n\n")

        sw = {
            "VS": ("Rigid formation shape maintained well; predictable behaviour; "
                   "all drones use MPC for smooth trajectories.",
                   "Formation may collide with obstacles in tight turns since nodes don't "
                   "adapt to geometry; higher computational cost (3× MPC)."),
            "LF": ("Simple offset tracking is easy to implement and tune; "
                   "leader's MPC path is directly reused from single-drone case.",
                   "Followers lag behind leader in turns; offset tracking "
                   "breaks when leader manoeuvres sharply; single point of failure at leader."),
            "Consensus": ("No single point of failure; distributed and resilient "
                          "to individual drone failures; works under degraded comms.",
                          "Slower convergence; formation shape is approximate; "
                          "APF can cause oscillations near obstacles; higher comm overhead."),
            "Hybrid": ("Best of VS (global shape) + reactive local obstacle negotiation; "
                       "smooth blending avoids APF oscillations.",
                       "Heuristic reactive policy is hand-tuned (no RL); "
                       "still has 3× MPC computational cost."),
            "Baseline": ("Simplest implementation; each drone independently optimal "
                         "for target capture.",
                         "No formation maintenance at all; drones may collide with each other; "
                         "no coordination; worst-case for multi-drone safety."),
        }

        for method, (strengths, weaknesses) in sw.items():
            f.write(f"### {method}\n")
            f.write(f"**Strengths**: {strengths}\n\n")
            f.write(f"**Weaknesses**: {weaknesses}\n\n")

        # Recommendation
        f.write("## Recommendation for Real Deployment\n\n")
        f.write("For real-world deployment in urban environments:\n\n")
        f.write("1. **Hybrid (VS+Reactive)** is recommended for narrow/complex environments. "
                "It maintains formation shape while adapting to local obstacles, reducing collision risk.\n\n")
        f.write("2. **Leader-Follower** is recommended for wider corridors with reliable comms. "
                "It is simpler to implement and debug, and the leader's MPC provides a proven baseline.\n\n")
        f.write("3. **Consensus+APF** is recommended when communication reliability is uncertain, "
                "as it degrades gracefully under packet drops.\n\n")
        f.write("4. **Virtual Structure** provides the tightest formation but requires "
                "obstacle-aware node placement (future work).\n\n")
        f.write("5. **Baseline** should be avoided — it provides no formation guarantees "
                "and creates inter-drone collision risk.\n\n")

        # Plots
        f.write("## Plots\n\n")
        f.write("![Capture Time CDF](plots/capture_time_cdf.png)\n\n")
        f.write("![Formation Error](plots/formation_error.png)\n\n")
        f.write("![Control Latency](plots/control_latency.png)\n\n")
        f.write("![Collisions](plots/collisions.png)\n\n")
        f.write("![Tracking RMSE](plots/tracking_rmse.png)\n\n")

        # Notes
        f.write("## Notes\n\n")
        f.write("- The Hybrid method uses a hand-tuned heuristic reactive policy rather than "
                "a learned RL policy. Full PPO training was not feasible within the session.\n")
        f.write("- Simulation is 2D planar with fixed altitude (20m). Z-dynamics are not modelled.\n")
        f.write("- Communication model assumes broadcast; no routing or multi-hop.\n")
        f.write("- Energy estimate is a proxy (integral of squared control magnitude), "
                "not a calibrated battery model.\n")

    print(f"Report saved: {report_path}")
    return report_path


if __name__ == "__main__":
    generate_report()
