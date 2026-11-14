import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import os

def compute_cost_matrix(drone_states, target_states, batteries, max_speed=15.0):
    """
    Computes the assignment cost matrix for N drones and M targets.
    Cost combines ETA (distance/speed) and battery life penalty.
    """
    N = len(drone_states)
    M = len(target_states)
    cost_matrix = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            # 1. ETA Cost (Distance / Max Speed)
            dist = np.linalg.norm(drone_states[i, 0:2] - target_states[j, 0:2])
            eta = dist / max_speed
            
            # 2. Battery Cost (Penalize lower battery)
            # Drones with lower battery should take targets that are closer
            # We add a penalty proportional to the consumed battery
            batt_penalty = (100.0 - batteries[i]) * 0.05
            
            # Optional: Add a line-of-sight penalty or other metrics here
            
            cost_matrix[i, j] = eta + batt_penalty
            
    return cost_matrix

def run_split_scenario():
    print("--- Multi-Drone Cooperative Target Assignment ---")
    print("Scenario: 3 Hummingbirds chasing 1 suspect. Suspect reaches an intersection and splits into 3.")
    
    # 1. Drone Fleet Setup
    # Drones in a loose triangle formation, moving right (+x)
    drone_states = np.array([
        [90.0,  0.0, 5.0, 0.0],  # Drone 0 (Leader, front)
        [85.0,  5.0, 5.0, 0.0],  # Drone 1 (Follower, left)
        [85.0, -5.0, 5.0, 0.0]   # Drone 2 (Follower, right)
    ])
    
    # Battery states (Drone 2 is almost depleted, Drone 1 is fresh)
    batteries = np.array([50.0, 5.0, 15.0])
    
    print("\n[Drone Status]")
    for i in range(3):
        print(f"Drone {i}: Pos=({drone_states[i,0]:.1f}, {drone_states[i,1]:.1f}), Batt={batteries[i]:.1f}%")
        
    # 2. Target Split Event
    # Suspect reaches the junction at x=100 and splits into 3 individuals
    intersection_x = 100.0
    
    target_states = np.array([
        [102.0,   0.0,  5.0,  0.0],  # Suspect 0: Continues straight (+x)
        [100.0,   2.0,  0.0,  5.0],  # Suspect 1: Turns left (+y)
        [100.0,  -2.0,  0.0, -5.0]   # Suspect 2: Turns right (-y)
    ])
    
    print("\n[Target Split Detected!]")
    for j in range(3):
        print(f"Target {j}: Pos=({target_states[j,0]:.1f}, {target_states[j,1]:.1f})")
    
    # 3. Compute Cost Matrix
    cost_matrix = compute_cost_matrix(drone_states, target_states, batteries)
    
    print("\n[Cost Matrix] (Rows: Drones, Cols: Targets)")
    print("      T0      T1      T2")
    for i in range(3):
        print(f"D{i} | {cost_matrix[i,0]:.2f} | {cost_matrix[i,1]:.2f} | {cost_matrix[i,2]:.2f} |")
        
    # 4. Hungarian Algorithm Assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    print("\n[Hungarian Algorithm Assignment Results]")
    for i, j in zip(row_ind, col_ind):
        dist = np.linalg.norm(drone_states[i, 0:2] - target_states[j, 0:2])
        print(f"-> Drone {i} (Batt: {batteries[i]:.0f}%) ASSIGNED TO Target {j} (Dist: {dist:.1f}m, Cost: {cost_matrix[i, j]:.2f})")
        
    # 5. Visualize Assignment
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Drones
    cmap = plt.get_cmap('tab10')
    for i in range(3):
        ax.scatter(drone_states[i, 0], drone_states[i, 1],
                   c=[cmap(i)], s=80, edgecolors='k', zorder=3,
                   label=f'D{i} ({batteries[i]:.0f}%)')
        
    # Plot Targets
    for j in range(3):
        ax.scatter(target_states[j, 0], target_states[j, 1],
                   c='red', marker='o', s=150, edgecolors='darkred', zorder=3)
        
    # Plot Assignments (Lines)
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
    out_path = os.path.join('results', 'assignment_split.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {out_path}")

if __name__ == "__main__":
    run_split_scenario()
