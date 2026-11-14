import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import os

def compute_cost_matrix(drone_states, target_states, batteries, max_speed=15.0):
    """
    Computes the assignment cost matrix for N drones and M targets.
    Cost combines ETA (distance/speed) and battery life penalty.
    Handles rectangular matrices (N != M).
    """
    N = len(drone_states)
    M = len(target_states)
    cost_matrix = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            dist = np.linalg.norm(drone_states[i, 0:2] - target_states[j, 0:2])
            eta = dist / max_speed
            batt_penalty = (100.0 - batteries[i]) * 0.05
            cost_matrix[i, j] = eta + batt_penalty
            
    return cost_matrix

def run_v_formation_split2():
    print("=" * 60)
    print("  V-Formation Drones vs Suspect Splitting into 2")
    print("=" * 60)
    print("Scenario: 5 drones in V-formation chasing 1 suspect.")
    print("Suspect splits into 2 at a fork. Hungarian algorithm")
    print("selects the best 2 drones out of 5 (rectangular matrix).")
    
    # 1. Drone Fleet: V-formation pointing forward (+x)
    #
    #        D0            (tip of V)
    #      D1  D2          (inner wings)
    #    D3      D4        (outer wings)
    #
    drone_states = np.array([
        [85.0,   0.0, 5.0, 0.0],  # D0: Tip (leader)
        [80.0,   6.0, 5.0, 0.0],  # D1: Inner left
        [80.0,  -6.0, 5.0, 0.0],  # D2: Inner right
        [75.0,  12.0, 5.0, 0.0],  # D3: Outer left
        [75.0, -12.0, 5.0, 0.0],  # D4: Outer right
    ])
    
    # Battery states: varied levels
    batteries = np.array([70.0, 45.0, 90.0, 30.0, 60.0])
    
    print("\n[Drone Status]")
    for i in range(len(drone_states)):
        print(f"  Drone {i}: Pos=({drone_states[i,0]:6.1f}, {drone_states[i,1]:6.1f}), "
              f"Batt={batteries[i]:.0f}%")
        
    # 2. Target Split: suspect forks into 2 at a Y-junction
    target_states = np.array([
        [105.0,  10.0,  4.0,  3.0],  # T0: Forks upper-right
        [105.0, -10.0,  4.0, -3.0],  # T1: Forks lower-right
    ])
    
    print("\n[Target Split Detected!]")
    for j in range(len(target_states)):
        print(f"  Target {j}: Pos=({target_states[j,0]:6.1f}, {target_states[j,1]:6.1f})")
    
    # 3. Compute Cost Matrix (5 drones x 2 targets => rectangular)
    cost_matrix = compute_cost_matrix(drone_states, target_states, batteries)
    
    print(f"\n[Cost Matrix] ({len(drone_states)} Drones x {len(target_states)} Targets)")
    header = "       " + "    ".join([f"  T{j}  " for j in range(len(target_states))])
    print(header)
    for i in range(len(drone_states)):
        row_str = f"  D{i} | " + " | ".join([f"{cost_matrix[i,j]:5.2f}" for j in range(len(target_states))]) + " |"
        print(row_str)
        
    # 4. Hungarian Algorithm (handles rectangular: picks best 2 from 5)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    total_cost = cost_matrix[row_ind, col_ind].sum()
    
    print("\n[Hungarian Algorithm Assignment Results]")
    print("-" * 50)
    assigned_drones = set(row_ind)
    for i, j in zip(row_ind, col_ind):
        dist = np.linalg.norm(drone_states[i, 0:2] - target_states[j, 0:2])
        print(f"  -> Drone {i} (Batt: {batteries[i]:.0f}%) => Target {j}  "
              f"[Dist: {dist:.1f}m, Cost: {cost_matrix[i, j]:.2f}]")
    
    unassigned = [i for i in range(len(drone_states)) if i not in assigned_drones]
    print(f"\n  Unassigned drones (available as backup): {unassigned}")
    print(f"  Total assignment cost: {total_cost:.2f}")
        
    # 5. Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = plt.get_cmap('tab10')
    
    # Draw V-formation lines (faint)
    v_order = [3, 1, 0, 2, 4]  # outer-left -> inner-left -> tip -> inner-right -> outer-right
    vx = [drone_states[i, 0] for i in v_order]
    vy = [drone_states[i, 1] for i in v_order]
    ax.plot(vx, vy, '-', color='gray', linewidth=1.0, alpha=0.4, zorder=1)
    
    # Plot drones
    for i in range(len(drone_states)):
        marker_alpha = 1.0 if i in assigned_drones else 0.35
        ax.scatter(drone_states[i, 0], drone_states[i, 1],
                   c=[cmap(i)], s=80, edgecolors='k', zorder=3, alpha=marker_alpha,
                   label=f'D{i} ({batteries[i]:.0f}%)')
        
    # Plot targets
    for j in range(len(target_states)):
        ax.scatter(target_states[j, 0], target_states[j, 1],
                   c='red', marker='o', s=150, edgecolors='darkred', zorder=3)
        
    # Plot assignment lines
    for i, j in zip(row_ind, col_ind):
        ax.plot([drone_states[i, 0], target_states[j, 0]], 
                [drone_states[i, 1], target_states[j, 1]], 
                color=cmap(i), linestyle='--', linewidth=2.5, zorder=2)
                 
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    os.makedirs('results', exist_ok=True)
    out_path = os.path.join('results', 'assignment_v_split2.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {out_path}")

if __name__ == "__main__":
    run_v_formation_split2()
