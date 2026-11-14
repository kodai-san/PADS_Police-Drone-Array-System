"""
Plotting module — generates comparison plots from aggregated results.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


def load_aggregated(filepath: str):
    """Load aggregated_results.csv into a list of dicts."""
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
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
    return rows


def plot_capture_time_cdf(rows, output_dir):
    """Capture time CDF per method."""
    methods = sorted(set(r["method"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:
        times = [r["capture_time"] for r in rows
                 if r["method"] == method and r["capture_time"] is not None]
        if not times:
            continue
        times.sort()
        n = len(times)
        cdf = np.arange(1, n + 1) / n
        ax.step(times, cdf, where='post', linewidth=2, label=method)

    ax.set_xlabel("Capture Time (s)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("Capture Time CDF by Method", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "capture_time_cdf.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_formation_error(rows, output_dir):
    """Formation error over methods (box plot / bar chart with CI)."""
    methods = sorted(set(r["method"] for r in rows))
    experiments = sorted(set(r["experiment"] for r in rows))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.18
    colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))

    for idx, exp in enumerate(experiments):
        means = []
        stds = []
        for method in methods:
            vals = [r["formation_error_mean"] for r in rows
                    if r["method"] == method and r["experiment"] == exp
                    and r["formation_error_mean"] is not None]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + idx * width, means, width, yerr=stds,
               label=exp, color=colors[idx], capsize=3)

    ax.set_xticks(x + width * (len(experiments) - 1) / 2)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Formation Error (m)", fontsize=12)
    ax.set_title("Mean Formation Error by Method and Experiment", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    path = os.path.join(output_dir, "formation_error.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_control_latency(rows, output_dir):
    """Control latency distribution per method."""
    methods = sorted(set(r["method"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 5))

    data = []
    labels = []
    for method in methods:
        vals = [r["control_latency_mean_ms"] for r in rows
                if r["method"] == method and r["control_latency_mean_ms"] is not None]
        if vals:
            data.append(vals)
            labels.append(method)

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    ax.set_ylabel("Control Latency (ms)", fontsize=12)
    ax.set_title("Control Computation Time Distribution", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    path = os.path.join(output_dir, "control_latency.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_collisions(rows, output_dir):
    """Collision count bar chart per method × experiment."""
    methods = sorted(set(r["method"] for r in rows))
    experiments = sorted(set(r["experiment"] for r in rows))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.18
    colors = plt.cm.Set1(np.linspace(0, 1, len(experiments)))

    for idx, exp in enumerate(experiments):
        counts = []
        for method in methods:
            vals = [r["collisions_count"] for r in rows
                    if r["method"] == method and r["experiment"] == exp
                    and r["collisions_count"] is not None]
            counts.append(np.mean(vals) if vals else 0)
        ax.bar(x + idx * width, counts, width,
               label=exp, color=colors[idx])

    ax.set_xticks(x + width * (len(experiments) - 1) / 2)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Mean Collision Count", fontsize=12)
    ax.set_title("Collisions by Method and Experiment", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    path = os.path.join(output_dir, "collisions.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_tracking_rmse(rows, output_dir):
    """Tracking RMSE comparison."""
    methods = sorted(set(r["method"] for r in rows))
    experiments = sorted(set(r["experiment"] for r in rows))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.18
    colors = plt.cm.Set3(np.linspace(0, 1, len(experiments)))

    for idx, exp in enumerate(experiments):
        means = []
        stds = []
        for method in methods:
            vals = [r["tracking_RMSE"] for r in rows
                    if r["method"] == method and r["experiment"] == exp
                    and r["tracking_RMSE"] is not None]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        ax.bar(x + idx * width, means, width, yerr=stds,
               label=exp, color=colors[idx], capsize=3)

    ax.set_xticks(x + width * (len(experiments) - 1) / 2)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Tracking RMSE (m)", fontsize=12)
    ax.set_title("Tracking RMSE by Method", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    path = os.path.join(output_dir, "tracking_rmse.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {path}")


def generate_all_plots(results_dir: str = "results"):
    """Generate all comparison plots from aggregated CSV."""
    agg_path = os.path.join(results_dir, "aggregated_results.csv")
    if not os.path.exists(agg_path):
        print(f"Error: {agg_path} not found. Run experiment_runner.py first.")
        return

    plot_dir = os.path.join(results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    rows = load_aggregated(agg_path)
    print(f"Loaded {len(rows)} trial results")

    plot_capture_time_cdf(rows, plot_dir)
    plot_formation_error(rows, plot_dir)
    plot_control_latency(rows, plot_dir)
    plot_collisions(rows, plot_dir)
    plot_tracking_rmse(rows, plot_dir)

    print(f"\nAll plots saved to {plot_dir}")


if __name__ == "__main__":
    generate_all_plots()
