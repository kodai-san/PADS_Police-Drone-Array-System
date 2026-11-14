"""
Synthetic corridor environment: T-shaped junction.

The corridor is built from wall polygons (rectangles) so the same
point-in-polygon / segment-intersection logic from the MATLAB code applies.
"""
import numpy as np
from typing import List, Tuple


def _rect_poly(cx, cy, hw, hh, angle=0.0) -> np.ndarray:
    """Axis-aligned rectangle as 4×2 polygon, optionally rotated."""
    corners = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ])
    if abs(angle) > 1e-9:
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        corners = corners @ R.T
    corners[:, 0] += cx
    corners[:, 1] += cy
    return corners


def create_corridor(length: float = 300.0,
                    width: float = 12.0,
                    turn_pos: float = 150.0,
                    turn_width: float = 12.0,
                    wall_thick: float = 3.0) -> dict:
    """
    Build a T-shaped corridor junction with thin wall outlines.

    Layout (top-down view, corridor is the open space):

        ═══════════════════════════════════   ← top wall
        ┃  approach corridor (along +X)   ┃
        ══════════╗          ╔════════════   ← bottom walls with gap
                  ┃ (branch) ┃
                  ┃  along   ┃
                  ┃   -Y     ┃
                  ╚══════════╝            ← branch bottom cap

    The horizontal top-bar extends from x=0 to x=length.
    A perpendicular branch drops from y=-hw downward at x=turn_pos.
    The target enters from the left, reaches the junction, and turns
    into the downward branch.

    Returns dict with:
        obstacles: list of Nx2 polygon arrays (thin walls)
        waypoints: Px2 array of corridor centre-line waypoints
        bounds:    [xmin, xmax, ymin, ymax]
    """
    hw = width / 2.0           # half-width of horizontal corridor
    tw = turn_width / 2.0      # half-width of the branch
    wt = wall_thick            # thin walls

    top_bar_len = length       # horizontal bar total length
    branch_len = length - turn_pos  # branch going downward

    obstacles: List[np.ndarray] = []

    # ── 1. Top wall of horizontal bar (full length) ──────────────────
    #    Thin rectangle sitting just above the corridor
    obstacles.append(_rect_poly(
        top_bar_len / 2, hw + wt / 2,
        top_bar_len / 2, wt / 2))

    # ── 2. Bottom wall LEFT of junction ──────────────────────────────
    #    From x=0 to x = turn_pos - tw (left of the branch opening)
    left_end = turn_pos - tw
    if left_end > 0:
        obstacles.append(_rect_poly(
            left_end / 2, -(hw + wt / 2),
            left_end / 2, wt / 2))

    # ── 3. Bottom wall RIGHT of junction ─────────────────────────────
    #    From x = turn_pos + tw to x = top_bar_len
    right_start = turn_pos + tw
    right_len = top_bar_len - right_start
    if right_len > 0:
        obstacles.append(_rect_poly(
            right_start + right_len / 2, -(hw + wt / 2),
            right_len / 2, wt / 2))

    # ── 4. Left wall of downward branch ──────────────────────────────
    obstacles.append(_rect_poly(
        turn_pos - tw - wt / 2, -(hw + branch_len / 2),
        wt / 2, branch_len / 2))

    # ── 5. Right wall of downward branch ─────────────────────────────
    obstacles.append(_rect_poly(
        turn_pos + tw + wt / 2, -(hw + branch_len / 2),
        wt / 2, branch_len / 2))

    # ── 6. Bottom cap of branch ──────────────────────────────────────
    obstacles.append(_rect_poly(
        turn_pos, -(hw + branch_len + wt / 2),
        tw + wt, wt / 2))

    # ── Target waypoints: enter from left, turn down into branch ─────
    n_approach = max(int(turn_pos), 20)
    n_branch = max(int(branch_len), 20)

    # Approach: along +X from x=0 to x=turn_pos, y=0
    wp1_x = np.linspace(0, turn_pos, n_approach, endpoint=False)
    wp1_y = np.zeros_like(wp1_x)

    # Branch: along -Y from y=0 down to y=-branch_len, at x=turn_pos
    wp2_y = np.linspace(0, -branch_len, n_branch)
    wp2_x = np.full_like(wp2_y, turn_pos)

    waypoints = np.column_stack([
        np.concatenate([wp1_x, wp2_x]),
        np.concatenate([wp1_y, wp2_y]),
    ])

    margin = 10.0
    xmin = -margin
    xmax = top_bar_len + margin
    ymin = -(hw + branch_len + wt) - margin
    ymax = hw + wt + margin
    bounds = [xmin, xmax, ymin, ymax]

    return {
        "obstacles": obstacles,
        "waypoints": waypoints,
        "bounds": bounds,
        "width": width,
        "turn_width": turn_width,
    }


# ── Geometry helpers ────────────────────────────────────────────────────
def point_in_poly(p: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = p[0], p[1]
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def point_in_any_obstacle(p: np.ndarray, obstacles: list) -> bool:
    for obs in obstacles:
        if point_in_poly(p, obs):
            return True
    return False


def _segments_intersect(a1, a2, b1, b2) -> bool:
    """Check if line segments a1-a2 and b1-b2 intersect."""
    d1 = a2 - a1
    d2 = b2 - b1
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return False
    t = ((b1[0] - a1[0]) * d2[1] - (b1[1] - a1[1]) * d2[0]) / cross
    u = ((b1[0] - a1[0]) * d1[1] - (b1[1] - a1[1]) * d1[0]) / cross
    return 0 <= t <= 1 and 0 <= u <= 1


def segment_intersects_any_obstacle(p1: np.ndarray, p2: np.ndarray,
                                     obstacles: list) -> bool:
    for obs in obstacles:
        n = len(obs)
        for i in range(n):
            j = (i + 1) % n
            if _segments_intersect(p1, p2, obs[i], obs[j]):
                return True
    return False


def inflate_obstacles(obstacles: list, r: float) -> list:
    """Push each vertex outward from polygon centroid by r metres."""
    inflated = []
    for obs in obstacles:
        cx = np.mean(obs[:, 0])
        cy = np.mean(obs[:, 1])
        dx = obs[:, 0] - cx
        dy = obs[:, 1] - cy
        dist = np.sqrt(dx**2 + dy**2) + 1e-12
        new_obs = obs.copy()
        new_obs[:, 0] += r * dx / dist
        new_obs[:, 1] += r * dy / dist
        inflated.append(new_obs)
    return inflated
