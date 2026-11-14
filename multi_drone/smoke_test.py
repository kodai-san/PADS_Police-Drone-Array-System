"""
Smoke test — 10-second validation of the MPC wrapper and simulation pipeline.
"""
import time
import numpy as np
import sys

from config import SimConfig
from dynamics import make_AB, propagate_drone
from mpc_solver import MPCSolver, build_prediction_matrices, diff_matrix_2d
from environment import create_corridor, point_in_any_obstacle
from target import target_step_waypoint


def test_mpc_solver():
    """Test that MPC solver produces valid output and timing."""
    print("=" * 50)
    print("  SMOKE TEST: MPC Solver")
    print("=" * 50)

    dt = 0.05
    A, B = make_AB(dt)
    N = 30
    solver = MPCSolver(A, B, N,
                       w_pos=15.0, w_vel=0.1, w_u=2.93, w_du=3.13,
                       amax=5.0)

    state = np.array([0.0, 0.0, 0.0, 0.0])
    u_prev = np.array([0.0, 0.0])

    # Target 10m ahead moving at 5 m/s
    pref = np.zeros((2, N + 1))
    vref = np.zeros((2, N + 1))
    for i in range(N + 1):
        pref[0, i] = 10.0 + 5.0 * i * dt
        pref[1, i] = 0.0
        vref[0, i] = 5.0
        vref[1, i] = 0.0

    # Solve
    U, solve_ms = solver.solve(state, u_prev, pref, vref)

    assert U is not None, "MPC solve returned None!"
    assert not np.any(np.isnan(U)), "MPC solution contains NaN!"
    assert solve_ms < 500.0, f"MPC solve too slow: {solve_ms:.1f} ms"

    print(f"  ✓ MPC solved in {solve_ms:.1f} ms")
    print(f"  ✓ First accel: [{U[0]:.3f}, {U[1]:.3f}] m/s²")
    print(f"  ✓ Accel magnitude: {np.linalg.norm(U[0:2]):.3f} m/s² (max=5.0)")

    return True


def test_corridor():
    """Test corridor creation and collision detection."""
    print("\n" + "=" * 50)
    print("  SMOKE TEST: Corridor Environment")
    print("=" * 50)

    env = create_corridor()
    assert len(env["obstacles"]) > 0, "No obstacles created!"
    assert len(env["waypoints"]) > 10, "Too few waypoints!"
    assert env["bounds"][0] < env["bounds"][1], "Invalid bounds!"

    # Centre should be clear
    assert not point_in_any_obstacle(np.array([50.0, 0.0]), env["obstacles"]), \
        "Centre of corridor should be clear!"
    # Inside the top wall should be blocked (top wall at y=6 to y=9 for default width=12, wall_thick=3)
    assert point_in_any_obstacle(np.array([50.0, 7.5]), env["obstacles"]), \
        "Inside wall should be blocked!"
    # Just inside corridor should be clear
    assert not point_in_any_obstacle(np.array([50.0, 5.0]), env["obstacles"]), \
        "Inside corridor edge should be clear!"

    print(f"  ✓ Corridor created: {len(env['obstacles'])} obstacles, "
          f"{len(env['waypoints'])} waypoints")
    print(f"  ✓ Collision detection works")

    return True


def test_10s_simulation():
    """Run a 10-second simulation and verify basic sanity."""
    print("\n" + "=" * 50)
    print("  SMOKE TEST: 10s Simulation")
    print("=" * 50)

    cfg = SimConfig()
    cfg.t_max = 10.0
    cfg.save_video = False

    from controllers.baseline import BaselineController
    controller = BaselineController(cfg)

    from simulator import run_trial
    t0 = time.perf_counter()
    result = run_trial(controller, cfg, seed=42, save_video=False)
    elapsed = time.perf_counter() - t0

    log = result["log"]
    metrics = result["metrics"]

    assert len(log["t"]) > 10, f"Too few steps: {len(log['t'])}"
    assert not np.any(np.isnan(log["drone_states"])), "NaN in drone states!"
    assert not np.any(np.isnan(log["target_states"])), "NaN in target states!"

    # Drones should have moved from initial position
    initial_pos = log["drone_states"][0, 0, 0:2]
    final_pos = log["drone_states"][-1, 0, 0:2]
    dist_moved = np.linalg.norm(final_pos - initial_pos)
    assert dist_moved > 1.0, f"Drone barely moved: {dist_moved:.2f} m"

    print(f"  ✓ Simulated {len(log['t'])} steps in {elapsed:.1f}s")
    print(f"  ✓ Drone moved {dist_moved:.1f} m")
    print(f"  ✓ Tracking RMSE: {metrics['tracking_RMSE']:.2f} m")
    print(f"  ✓ Control latency: {metrics['control_latency_mean_ms']:.1f} ms avg, "
          f"{metrics['control_latency_95pct_ms']:.1f} ms 95th pct")
    print(f"  ✓ Collisions: {metrics['collisions_count']}")
    print(f"  ✓ Wall time: {elapsed:.1f}s for 10s simulation "
          f"({elapsed/10.0:.1f}x real-time)")

    return True


def main():
    print("\n🚀 Running Multi-Drone MPC Smoke Tests\n")
    all_pass = True

    try:
        all_pass &= test_mpc_solver()
    except Exception as e:
        print(f"  ✗ MPC solver test FAILED: {e}")
        all_pass = False

    try:
        all_pass &= test_corridor()
    except Exception as e:
        print(f"  ✗ Corridor test FAILED: {e}")
        all_pass = False

    try:
        all_pass &= test_10s_simulation()
    except Exception as e:
        print(f"  ✗ 10s simulation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False

    print("\n" + "=" * 50)
    if all_pass:
        print("  ✅ ALL SMOKE TESTS PASSED")
    else:
        print("  ❌ SOME TESTS FAILED")
    print("=" * 50)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
