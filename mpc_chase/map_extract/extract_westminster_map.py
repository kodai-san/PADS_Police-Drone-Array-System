"""
extract_westminster_map.py

Fully offline map extraction — no network required.
Reads buildings and roads from local GeoJSON files exported from overpass-turbo.eu.

Setup:
  Go to https://overpass-turbo.eu and run TWO queries:

  BUILDINGS — paste, Run, Export -> GeoJSON, save as buildings.geojson:

    [out:json][timeout:30];
    (
      way["building"](51.4938,-0.1340,51.4975,-0.1270);
    );
    out body;
    >;
    out skel qt;

  ROADS — paste, Run, Export -> GeoJSON, save as roads.geojson:

    [out:json][timeout:30];
    (
      way["highway"~"primary|secondary|tertiary|residential|footway|path|pedestrian|service|unclassified|trunk|living_street"](51.4938,-0.1340,51.4975,-0.1270);
    );
    out body;
    >;
    out skel qt;

Requirements:
    py -m pip install scipy numpy networkx

Usage:
    py extract_westminster_map.py

Output:
    westminster_map.mat
"""

import numpy as np
from scipy.io import savemat
import networkx as nx
import json
import os
import sys

# ============================================================
#  Config — Victoria St / Horseferry Rd area (real street grid)
# ============================================================
SOUTH, NORTH = 51.4938, 51.4975
WEST, EAST   = -0.1340, -0.1270

LAT_REF = 51.4955
LON_REF = -0.1305
R_EARTH = 6371000.0

BUILDING_FILE = "buildings.geojson"
ROAD_FILE     = "roads.geojson"

def latlon_to_enu(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    east  = R_EARTH * np.cos(np.radians(LAT_REF)) * np.radians(lon - LON_REF)
    north = R_EARTH * np.radians(lat - LAT_REF)
    return east, north

def check_file(fname):
    if not os.path.exists(fname):
        print(f"ERROR: '{fname}' not found in {os.getcwd()}")
        print(f"Download it from overpass-turbo.eu (see instructions at top of script)")
        sys.exit(1)

# ============================================================
#  1. Buildings
# ============================================================
check_file(BUILDING_FILE)
print(f"Loading buildings from {BUILDING_FILE}...")

with open(BUILDING_FILE, "r", encoding="utf-8") as f:
    geojson = json.load(f)

building_polys = []
for feature in geojson.get("features", []):
    geom = feature.get("geometry", {})
    gtype = geom.get("type", "")

    coord_rings = []
    if gtype == "Polygon":
        coord_rings = [geom["coordinates"][0]]
    elif gtype == "MultiPolygon":
        for poly_coords in geom["coordinates"]:
            coord_rings.append(poly_coords[0])
    else:
        continue

    for ring in coord_rings:
        if len(ring) < 4:
            continue
        coords = np.array(ring)  # [lon, lat] per row
        easts, norths = latlon_to_enu(coords[:, 1], coords[:, 0])
        verts = np.column_stack([easts, norths])
        if np.allclose(verts[0], verts[-1], atol=0.01):
            verts = verts[:-1]
        if len(verts) >= 3:
            building_polys.append(verts)

print(f"  {len(building_polys)} buildings")

# ============================================================
#  2. Roads
# ============================================================
check_file(ROAD_FILE)
print(f"Loading roads from {ROAD_FILE}...")

with open(ROAD_FILE, "r", encoding="utf-8") as f:
    road_geojson = json.load(f)

# Build a graph from road LineStrings
G = nx.Graph()
node_id_counter = 0
coord_to_node = {}  # (rounded lat,lon) -> node_id
node_enu_map = {}    # node_id -> (east, north)

def get_or_create_node(lat, lon):
    global node_id_counter
    key = (round(lat, 6), round(lon, 6))
    if key in coord_to_node:
        return coord_to_node[key]
    nid = node_id_counter
    node_id_counter += 1
    coord_to_node[key] = nid
    e, n = latlon_to_enu(lat, lon)
    node_enu_map[nid] = (float(e), float(n))
    G.add_node(nid, x=float(e), y=float(n))
    return nid

road_segments = []

# Debug: inspect structure
top_keys = list(road_geojson.keys())
print(f"  GeoJSON top-level keys: {top_keys}")

features = road_geojson.get("features", [])
print(f"  {len(features)} features found")

# Show first feature for debugging
if features:
    f0 = features[0]
    g0 = f0.get("geometry", {})
    print(f"  First feature geometry type: {g0.get('type', 'NONE')}")
    coords0 = g0.get("coordinates", [])
    if coords0:
        print(f"  First coord sample: {coords0[0] if not isinstance(coords0[0], list) else coords0[0][:3]}")

geom_types_seen = set()

for feature in features:
    geom = feature.get("geometry", None)
    if geom is None:
        continue
    gtype = geom.get("type", "")
    geom_types_seen.add(gtype)

    lines = []
    if gtype == "LineString":
        lines = [geom["coordinates"]]
    elif gtype == "MultiLineString":
        lines = geom["coordinates"]
    elif gtype == "Polygon":
        # Some highway areas export as polygons — use outer ring as a line
        lines = [geom["coordinates"][0]]
    elif gtype == "Point":
        continue
    else:
        continue

    for line in lines:
        if len(line) < 2:
            continue
        prev_nid = None
        for coord in line:
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                lon, lat = float(coord[0]), float(coord[1])
            else:
                continue
            nid = get_or_create_node(lat, lon)
            if prev_nid is not None and prev_nid != nid:
                e1, n1 = node_enu_map[prev_nid]
                e2, n2 = node_enu_map[nid]
                dist = np.sqrt((e2 - e1)**2 + (n2 - n1)**2)
                G.add_edge(prev_nid, nid, length=dist)
                road_segments.append([e1, n1, e2, n2])
            prev_nid = nid

print(f"  Geometry types seen: {geom_types_seen}")
road_segments = np.array(road_segments) if road_segments else np.zeros((0, 4))
nodes_array = np.array(list(node_enu_map.values())) if node_enu_map else np.zeros((0, 2))
print(f"  {len(road_segments)} road segments, {len(nodes_array)} nodes")

# ============================================================
#  3. Target path (waypoints near Parliament Square)
# ============================================================
print("Computing target path...")

waypoint_latlon = np.array([
    [51.4965, -0.1330],   # Victoria St (west end)
    [51.4956, -0.1308],
    [51.4949, -0.1307],
    [51.4949, -0.1291],   # Horseferry Rd (south)
])

# Find nearest graph node to each waypoint
def nearest_node(lat, lon):
    if not node_enu_map:
        return None
    e, n = latlon_to_enu(lat, lon)
    best_nid = None
    best_dist = float("inf")
    for nid, (ne, nn) in node_enu_map.items():
        d = (ne - float(e))**2 + (nn - float(n))**2
        if d < best_dist:
            best_dist = d
            best_nid = nid
    return best_nid

waypoint_nodes = []
waypoint_enu = []
for wlat, wlon in waypoint_latlon:
    nid = nearest_node(wlat, wlon)
    waypoint_nodes.append(nid)
    e, n = latlon_to_enu(wlat, wlon)
    waypoint_enu.append([float(e), float(n)])
waypoint_enu = np.array(waypoint_enu)

# Build path through graph, or fall back to straight waypoint lines
if len(node_enu_map) > 0 and None not in waypoint_nodes:
    full_path_nodes = []
    for i in range(len(waypoint_nodes) - 1):
        try:
            path = list(nx.shortest_path(G, waypoint_nodes[i], waypoint_nodes[i+1], weight="length"))
            if i > 0:
                path = path[1:]
            full_path_nodes.extend(path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            print(f"  WARNING: No path between waypoints {i} and {i+1}")
            full_path_nodes.append(waypoint_nodes[i+1])

    if len(full_path_nodes) >= 2:
        path_enu = np.array([list(node_enu_map[n]) for n in full_path_nodes])
    else:
        print("  WARNING: Graph path too short, using straight waypoint lines")
        path_enu = waypoint_enu.copy()
else:
    print("  WARNING: No road graph available, using straight waypoint lines")
    path_enu = waypoint_enu.copy()
path_len = np.sum(np.sqrt(np.sum(np.diff(path_enu, axis=0)**2, axis=1)))
print(f"  Path: {len(path_enu)} nodes, {path_len:.0f} m")

# ============================================================
#  4. Save
# ============================================================
print("Saving westminster_map.mat...")

bldg_cell = np.empty((len(building_polys),), dtype=object)
for i, b in enumerate(building_polys):
    bldg_cell[i] = b

savemat("westminster_map.mat", {
    "road_segments":    road_segments,
    "road_nodes":       nodes_array,
    "building_polys":   bldg_cell,
    "n_buildings":      len(building_polys),
    "target_path":      path_enu,
    "waypoints_enu":    waypoint_enu,
    "waypoints_latlon": waypoint_latlon,
    "ref_latlon":       np.array([LAT_REF, LON_REF]),
    "bbox_latlon":      np.array([NORTH, SOUTH, EAST, WEST]),
}, do_compression=True)

print(f"\nDone! westminster_map.mat")
print(f"  Roads:     {road_segments.shape[0]} segments")
print(f"  Buildings: {len(building_polys)}")
print(f"  Path:      {path_enu.shape[0]} pts, {path_len:.0f} m")