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

def run_complex_scenario():
    print("--- Multi-Drone Complex Target Assignment ---")
    print("Scenario: 6 Drones chasing a group that splits into 6 individuals.")
    
    # 1. Drone Fleet Setup
    # 6 Drones scattered in a larger formation
    drone_states = np.array([
        [80.0,   0.0, 5.0, 0.0],  # Drone 0 (Center front)
        [75.0,  10.0, 5.0, 0.0],  # Drone 1 (Left)
        [75.0, -10.0, 5.0, 0.0],  # Drone 2 (Right)
        [70.0,  20.0, 5.0, 0.0],  # Drone 3 (Far Left)
        [70.0, -20.0, 5.0, 0.0],  # Drone 4 (Far Right)
        [60.0,   0.0, 5.0, 0.0]   # Drone 5 (Rear Guard)
    ])
    
    # Battery states: Drones 1 and 3 have very low battery. Drone 5 has full battery.
    batteries = np.array([80.0, 15.0, 90.0, 5.0, 40.0, 100.0])
    
    print("\n[Drone Status]")
    for i in range(len(drone_states)):
        print(f"Drone {i}: Pos=({drone_states[i,0]:.1f}, {drone_states[i,1]:.1f}), Batt={batteries[i]:.1f}%")
        
    # 2. Target Split Event
    # 6 suspects scattering in a plaza or open field
    target_states = np.array([
        [110.0,   0.0,  6.0,  0.0],  # T0: Fast straight
        [105.0,  15.0,  2.0,  4.0],  # T1: Top-right
        [105.0, -15.0,  2.0, -4.0],  # T2: Bottom-right
        [95.0,   25.0, -1.0,  5.0],  # T3: Top-left
        [95.0,  -25.0, -1.0, -5.0],  # T4: Bottom-left
        [115.0,   5.0,  5.0,  2.0]   # T5: Right slightly up
    ])
    
    print("\n[Target Split Detected!]")
    for j in range(len(target_states)):
        print(f"Target {j}: Pos=({target_states[j,0]:.1f}, {target_states[j,1]:.1f})")
    
    # 3. Compute Cost Matrix
    cost_matrix = compute_cost_matrix(drone_states, target_states, batteries)
    
    print("\n[Cost Matrix] (Rows: Drones, Cols: Targets)")
    header = "      " + "      ".join([f"T{j}" for j in range(len(target_states))])
    print(header)
    for i in range(len(drone_states)):
        row_str = f"D{i} | " + " | ".join([f"{cost_matrix[i,j]:.2f}" for j in range(len(target_states))]) + " |"
        print(row_str)
        
    # 4. Hungarian Algorithm Assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    print("\n[Hungarian Algorithm Assignment Results]")
    for i, j in zip(row_ind, col_ind):
        dist = np.linalg.norm(drone_states[i, 0:2] - target_states[j, 0:2])
        print(f"-> Drone {i} (Batt: {batteries[i]:.0f}%) ASSIGNED TO Target {j} (Dist: {dist:.1f}m, Cost: {cost_matrix[i, j]:.2f})")
        
    # 5. Visualize Assignment
    plt.figure(figsize=(10, 10))
    plt.title("Complex Real-Time Target Assignment (6 Drones, 6 Targets)")
    
    # Plot Drones
    cmap = plt.get_cmap('tab10')
    for i in range(len(drone_states)):
        plt.scatter(drone_states[i, 0], drone_states[i, 1], c=[cmap(i)], s=150, edgecolors='k', label=f'Drone {i} ({batteries[i]:.0f}%)')
        
    # Plot Targets
    for j in range(len(target_states)):
        plt.scatter(target_states[j, 0], target_states[j, 1], c='red', marker='*', s=200, edgecolors='darkred')
        plt.text(target_states[j, 0]+1.5, target_states[j, 1]+1.0, f'T{j}', fontweight='bold')
        
    # Plot Assignments (Lines)
    for i, j in zip(row_ind, col_ind):
        plt.plot([drone_states[i, 0], target_states[j, 0]], 
                 [drone_states[i, 1], target_states[j, 1]], 
                 color=cmap(i), linestyle='--', linewidth=2.0)
                 
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    # Move legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    os.makedirs('results', exist_ok=True)
    out_path = os.path.join('results', 'assignment_complex.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {out_path}")

if __name__ == "__main__":
    run_complex_scenario()
