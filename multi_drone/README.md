# Multi-Drone Formation Chase: Comparative Experiments

## Quick Start

```bash
cd c:\Users\kodai\Documents\3yp\multi_drone

# 1. Install dependencies
pip install -r requirements.txt

# 2. Run smoke test (10s validation)
python smoke_test.py

# 3. Run a single quick trial (20s)
python experiment_runner.py --method VS --experiment nominal --seed 1 --duration 20

# 4. Run full experiment batch (all methods × experiments × seeds)
python experiment_runner.py

# 5. Generate comparison plots
python plotting.py

# 6. Generate results report
python report.py
```

## Methods

| Method | Description |
|--------|-------------|
| **VS** (Virtual Structure) | Rigid virtual body tracking target; all drones use MPC to track their assigned node |
| **LF** (Leader-Follower) | Leader uses MPC to chase target; followers maintain fixed offset to leader |
| **Consensus** (Consensus+APF) | No fixed leader; distributed consensus for formation centroid + APF for avoidance |
| **Hybrid** (VS+Reactive) | Virtual structure for global shape + heuristic reactive policy for local obstacles |
| **Baseline** | Independent MPC per drone, no formation constraints |

## Experiments

| Experiment | Description |
|-----------|-------------|
| `nominal` | Ideal comms, default noise (σ_pos=0.1m) |
| `comms_degraded` | 30% packet drop, 100ms latency |
| `high_noise` | Position noise σ=0.5m |
| `narrow_turn` | Corridor width reduced to 8m at turn |

Each experiment runs 5 seeds (1-5).

## Output Structure

```
results/
├── aggregated_results.csv
├── report.md
├── plots/
│   ├── capture_time_cdf.png
│   ├── formation_error.png
│   ├── control_latency.png
│   ├── collisions.png
│   └── tracking_rmse.png
├── VS/
│   ├── nominal/
│   │   ├── seed1/
│   │   │   ├── VS_nominal_seed1.mp4
│   │   │   ├── VS_nominal_seed1_log.csv
│   │   │   └── VS_nominal_seed1_metrics.json
│   │   └── ...
│   └── ...
└── ...
```

## CLI Options

```
python experiment_runner.py [OPTIONS]

  --method METHOD        Single method (VS, LF, Consensus, Hybrid, Baseline)
  --experiment EXPT      Single experiment (nominal, comms_degraded, ...)
  --seed SEED            Single seed number
  --duration SECONDS     Override sim duration
  --results-dir DIR      Output directory (default: results)
  --no-video             Skip video generation (faster)
```

## Relation to Single-Drone MPC

The MPC solver in `mpc_solver.py` is a direct Python port of the MATLAB
`solveMPC_QP.m`. It uses the same QP formulation with:
- Position/velocity tracking weights
- Input magnitude penalty
- Input smoothing (delta-u) penalty
- Component-wise acceleration bounds

The VS and LF controllers reuse this MPC for all drones. The leader's
reference is the target position (same as the single-drone case). Followers
get offset references.

## Notes

- Simulation is 2D (x,y plane) with fixed altitude (20m). The z-axis is
  logged but not actively controlled.
- The Hybrid method uses a hand-tuned heuristic reactive policy (not RL).
  Full PPO training was not feasible within the session.
- Video generation uses matplotlib rendering. Install `imageio[ffmpeg]`
  for faster MP4 encoding.
