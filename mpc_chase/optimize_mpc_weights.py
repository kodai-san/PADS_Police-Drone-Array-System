#!/usr/bin/env python3
"""
MPC Weight Optimization via Bayesian Optimisation
==================================================

Runs 50 simulations of the Westminster MPC chase to find optimal controller
weights (w_pos, w_vel, w_u, w_du) using Gaussian-Process-based Bayesian
optimisation (scikit-optimize).

Usage:
    pip install scikit-optimize matplotlib numpy
    python optimize_mpc_weights.py

Each iteration calls MATLAB headless to run one simulation.  Results are
printed as a table and plotted as convergence + parallel-coordinates figures.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
N_CALLS       = 100       # total evaluations (including initial random)
N_RANDOM      = 10       # initial random exploration points
MATLAB_CMD    = "matlab"  # adjust if matlab is not on PATH
SCRIPT_DIR    = Path(__file__).resolve().parent

WEIGHTS_FILE  = SCRIPT_DIR / "_opt_weights.json"
METRICS_FILE  = SCRIPT_DIR / "_opt_metrics.json"
RESULTS_CSV   = SCRIPT_DIR / "weight_optimization_results.csv"
CONVERGE_PNG  = SCRIPT_DIR / "weight_optimization_convergence.png"
RESULTS_PNG   = SCRIPT_DIR / "weight_optimization_results.png"

# Search space  (name, low, high)
SPACE = [
    ("w_pos", 1.0,  80.0),
    ("w_vel", 0.1,  10.0),
    ("w_u",   0.01,  5.0),
    ("w_du",  0.1,  15.0),
]

# Combined cost weights
COST_WEIGHTS = {
    "dist_error_rms": 0.40,
    "total_energy_J": 0.30,
    "jerk_rms":       0.20,
    "los_penalty":    0.1,   
}

# Reference values for normalisation (from default weights run).
# These are updated after the first baseline trial.
REF_VALS = {
    "dist_error_rms": 5.0,
    "total_energy_J": 40000.0,
    "jerk_rms":       50.0,
}

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def run_matlab_trial(w_pos: float, w_vel: float, w_u: float, w_du: float,
                     trial_num: int) -> dict:
    """Write weights, invoke MATLAB, read back metrics."""
    weights = {"w_pos": w_pos, "w_vel": w_vel, "w_u": w_u, "w_du": w_du}
    WEIGHTS_FILE.write_text(json.dumps(weights))

    # Remove stale metrics
    if METRICS_FILE.exists():
        METRICS_FILE.unlink()

    cmd = [
        MATLAB_CMD, "-batch",
        "cd('{}'); run_mpc_weights_trial".format(str(SCRIPT_DIR).replace("\\", "/")),
    ]

    print(f"  [Trial {trial_num:3d}/{N_CALLS}]  w_pos={w_pos:7.2f}  "
          f"w_vel={w_vel:5.2f}  w_u={w_u:5.3f}  w_du={w_du:6.2f}  ...", end="", flush=True)

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return {"success": False}
    except FileNotFoundError:
        print(f"\n  ERROR: '{MATLAB_CMD}' not found. Make sure MATLAB is on your PATH.")
        sys.exit(1)

    elapsed = time.time() - t0

    if not METRICS_FILE.exists():
        print(f"  FAILED ({elapsed:.0f}s)")
        if result.stderr:
            print(f"    MATLAB stderr: {result.stderr[:300]}")
        return {"success": False}

    metrics = json.loads(METRICS_FILE.read_text())
    if not metrics.get("success", False):
        print(f"  FAILED ({elapsed:.0f}s)")
        return {"success": False}

    print(f"  dist={metrics['dist_error_rms']:.3f}m  E={metrics['total_energy_J']:.0f}J  "
          f"jerk={metrics['jerk_rms']:.1f}  LoS={metrics['los_fraction']:.1%}  ({elapsed:.0f}s)")
    return metrics


def compute_cost(metrics: dict, ref: dict) -> float:
    """Compute normalised combined cost from metrics."""
    if not metrics.get("success", False):
        return 10.0  # penalty for failed runs

    dist_norm   = metrics["dist_error_rms"]  / ref["dist_error_rms"]
    energy_norm = metrics["total_energy_J"]  / ref["total_energy_J"]
    jerk_norm   = metrics["jerk_rms"]        / ref["jerk_rms"]
    los_penalty = 1.0 - metrics["los_fraction"]

    cost = (COST_WEIGHTS["dist_error_rms"] * dist_norm +
            COST_WEIGHTS["total_energy_J"] * energy_norm +
            COST_WEIGHTS["jerk_rms"]       * jerk_norm +
            COST_WEIGHTS["los_penalty"]    * los_penalty)
    return cost


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    try:
        from skopt import gp_minimize
        from skopt.space import Real
        from skopt.plots import plot_convergence
    except ImportError:
        print("Installing scikit-optimize ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "scikit-optimize", "matplotlib", "numpy"])
        from skopt import gp_minimize
        from skopt.space import Real
        from skopt.plots import plot_convergence

    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    print("=" * 72)
    print("  MPC WEIGHT OPTIMISATION — Bayesian Optimisation  (50 trials)")
    print("=" * 72)

    # --- Step 0: Run baseline (default weights) to set reference values ---
    print("\n>>> Baseline trial with default weights ...")
    base_metrics = run_matlab_trial(20.0, 2.0, 0.5, 3.0, trial_num=0)
    if base_metrics.get("success"):
        REF_VALS["dist_error_rms"] = max(base_metrics["dist_error_rms"], 0.1)
        REF_VALS["total_energy_J"] = max(base_metrics["total_energy_J"], 100.0)
        REF_VALS["jerk_rms"]       = max(base_metrics["jerk_rms"], 1.0)
        print(f"  Baseline refs: dist={REF_VALS['dist_error_rms']:.3f}m, "
              f"E={REF_VALS['total_energy_J']:.0f}J, jerk={REF_VALS['jerk_rms']:.1f}")
    else:
        print("  WARNING: Baseline failed, using hard-coded reference values.")

    # --- Storage for all trials ---
    all_trials = []
    trial_counter = [0]  # mutable counter

    def objective(params):
        w_pos, w_vel, w_u, w_du = params
        trial_counter[0] += 1

        metrics = run_matlab_trial(w_pos, w_vel, w_u, w_du, trial_counter[0])
        cost = compute_cost(metrics, REF_VALS)

        all_trials.append({
            "trial":  trial_counter[0],
            "w_pos":  w_pos,
            "w_vel":  w_vel,
            "w_u":    w_u,
            "w_du":   w_du,
            "cost":   cost,
            **{k: metrics.get(k, float("nan")) for k in
               ["dist_error_rms", "total_energy_J", "jerk_rms",
                "los_fraction", "oscillation_score", "safety_fraction"]},
        })
        return cost

    # --- Define search space ---
    dimensions = [Real(lo, hi, name=name) for name, lo, hi in SPACE]

    # --- Run optimisation ---
    print(f"\n>>> Starting Bayesian optimisation: {N_CALLS} trials "
          f"({N_RANDOM} random + {N_CALLS - N_RANDOM} guided)\n")

    result = gp_minimize(
        objective,
        dimensions,
        n_calls=N_CALLS,
        n_initial_points=N_RANDOM,
        acq_func="EI",          # Expected Improvement
        noise=1e-6,             # assume nearly deterministic sim
        random_state=42,
        verbose=False,
    )

    # --- Results ---
    best_w_pos, best_w_vel, best_w_u, best_w_du = result.x
    best_cost = result.fun

    print("\n" + "=" * 72)
    print("  OPTIMISATION COMPLETE")
    print("=" * 72)
    print(f"  Best cost:  {best_cost:.6f}")
    print(f"  Best weights:")
    print(f"    w_pos = {best_w_pos:.4f}")
    print(f"    w_vel = {best_w_vel:.4f}")
    print(f"    w_u   = {best_w_u:.4f}")
    print(f"    w_du  = {best_w_du:.4f}")
    print("=" * 72)

    # --- Print full table ---
    print(f"\n{'Trial':>5s}  {'w_pos':>8s}  {'w_vel':>7s}  {'w_u':>7s}  "
          f"{'w_du':>7s}  {'Cost':>8s}  {'DistErr':>8s}  {'Energy':>8s}  "
          f"{'Jerk':>8s}  {'LoS%':>6s}")
    print("-" * 90)
    for t in all_trials:
        print(f"{t['trial']:5d}  {t['w_pos']:8.2f}  {t['w_vel']:7.3f}  "
              f"{t['w_u']:7.4f}  {t['w_du']:7.3f}  {t['cost']:8.4f}  "
              f"{t['dist_error_rms']:8.3f}  {t['total_energy_J']:8.0f}  "
              f"{t['jerk_rms']:8.2f}  {t['los_fraction']*100:5.1f}%")

    # --- Save CSV ---
    with open(RESULTS_CSV, "w") as f:
        header = list(all_trials[0].keys())
        f.write(",".join(header) + "\n")
        for t in all_trials:
            f.write(",".join(str(t[k]) for k in header) + "\n")
    print(f"\nResults saved to: {RESULTS_CSV}")

    # --- Convergence plot ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    costs = [t["cost"] for t in all_trials]
    best_so_far = np.minimum.accumulate(costs)
    ax1.plot(range(1, len(costs) + 1), costs, "o-", alpha=0.5, label="Trial cost", color="#5B9BD5")
    ax1.plot(range(1, len(costs) + 1), best_so_far, "s-", label="Best so far",
             color="#ED7D31", linewidth=2)
    ax1.axhline(y=best_cost, color="green", linestyle="--", alpha=0.5,
                label=f"Optimum = {best_cost:.4f}")
    ax1.set_xlabel("Trial Number", fontsize=12)
    ax1.set_ylabel("Combined Cost (normalised)", fontsize=12)
    ax1.set_title("MPC Weight Optimisation — Convergence", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(str(CONVERGE_PNG), dpi=150)
    print(f"Convergence plot:  {CONVERGE_PNG}")

    # --- Results dashboard ---
    fig2, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Scatter: w_pos vs cost
    ax = axes[0, 0]
    ax.scatter([t["w_pos"] for t in all_trials], costs, c=range(len(costs)),
               cmap="viridis", s=40, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("w_pos"); ax.set_ylabel("Cost"); ax.set_title("w_pos vs Cost")
    ax.grid(True, alpha=0.3)

    # Scatter: w_vel vs cost
    ax = axes[0, 1]
    ax.scatter([t["w_vel"] for t in all_trials], costs, c=range(len(costs)),
               cmap="viridis", s=40, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("w_vel"); ax.set_ylabel("Cost"); ax.set_title("w_vel vs Cost")
    ax.grid(True, alpha=0.3)

    # Scatter: w_u vs cost
    ax = axes[0, 2]
    ax.scatter([t["w_u"] for t in all_trials], costs, c=range(len(costs)),
               cmap="viridis", s=40, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("w_u"); ax.set_ylabel("Cost"); ax.set_title("w_u vs Cost")
    ax.grid(True, alpha=0.3)

    # Scatter: w_du vs cost
    ax = axes[1, 0]
    ax.scatter([t["w_du"] for t in all_trials], costs, c=range(len(costs)),
               cmap="viridis", s=40, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("w_du"); ax.set_ylabel("Cost"); ax.set_title("w_du vs Cost")
    ax.grid(True, alpha=0.3)

    # Metric breakdown for top-10 trials
    ax = axes[1, 1]
    sorted_trials = sorted(all_trials, key=lambda t: t["cost"])[:10]
    indices = range(len(sorted_trials))
    ax.barh(indices, [t["dist_error_rms"] for t in sorted_trials],
            color="#5B9BD5", label="Dist Err RMS (m)")
    ax.set_yticks(indices)
    ax.set_yticklabels([f"#{t['trial']}" for t in sorted_trials])
    ax.set_xlabel("Distance Error RMS (m)")
    ax.set_title("Top-10 Trials — Distance Error")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # Parallel coordinates for top-10
    ax = axes[1, 2]
    param_names = ["w_pos", "w_vel", "w_u", "w_du"]
    # Normalise each param to [0, 1] for visualization
    param_ranges = {name: (lo, hi) for name, lo, hi in SPACE}
    for i, t in enumerate(sorted_trials):
        normed = [(t[p] - param_ranges[p][0]) / (param_ranges[p][1] - param_ranges[p][0])
                  for p in param_names]
        alpha = 1.0 - 0.06 * i
        ax.plot(range(len(param_names)), normed, "o-", alpha=alpha,
                linewidth=2.0 - 0.1 * i, label=f"#{t['trial']} (c={t['cost']:.3f})")
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names)
    ax.set_ylabel("Normalised Value")
    ax.set_title("Top-10 Trials — Weight Profiles")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig2.suptitle("MPC Weight Optimisation — Results Dashboard", fontsize=14, y=0.98)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(str(RESULTS_PNG), dpi=150)
    print(f"Results dashboard: {RESULTS_PNG}")

    # --- Highlight best vs default ---
    print(f"\n{'='*72}")
    print(f"  COMPARISON: Default vs Optimal Weights")
    print(f"{'='*72}")
    print(f"  {'':20s}  {'Default':>12s}  {'Optimal':>12s}")
    print(f"  {'w_pos':20s}  {'20.0':>12s}  {best_w_pos:12.4f}")
    print(f"  {'w_vel':20s}  {'2.0':>12s}  {best_w_vel:12.4f}")
    print(f"  {'w_u':20s}  {'0.5':>12s}  {best_w_u:12.4f}")
    print(f"  {'w_du':20s}  {'3.0':>12s}  {best_w_du:12.4f}")

    # Find the trial closest to default weights to compare metrics
    if base_metrics.get("success"):
        base_cost = compute_cost(base_metrics, REF_VALS)
        best_trial = sorted_trials[0]
        print(f"\n  {'Cost':20s}  {base_cost:12.4f}  {best_trial['cost']:12.4f}  "
              f"({'↓' if best_trial['cost'] < base_cost else '↑'} "
              f"{abs(1 - best_trial['cost']/base_cost)*100:.1f}%)")
    print(f"{'='*72}\n")

    # Cleanup temp files
    for f in [WEIGHTS_FILE, METRICS_FILE]:
        if f.exists():
            f.unlink()


if __name__ == "__main__":
    main()
