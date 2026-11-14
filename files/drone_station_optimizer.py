"""
Drone Station Placement Optimizer — Westminster Burglary Response
=================================================================
1. Reads burglary lon/lat from Excel columns E & F
2. Adds false alarm rate (Normal, mean=10%, std=2%) per incident
3. Weighted K-Means clustering to find optimal station locations
   (weight = 1 - false_alarm_rate -> real crimes pull harder)
4. Outputs:
   a) Map images for various station counts on Westminster crime density
   b) Table: avg response time vs number of stations
   c) Dedicated map for the config closest to 1.5 min response
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
import warnings, os
warnings.filterwarnings('ignore')
np.random.seed(42)

# -- paths --
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT  = os.path.join(BASE_DIR, "westminster_burglaries_filtered.xlsx")
OUTDIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTDIR, exist_ok=True)

# -- drone parameters --
T_UP    = 20.0    # ascend [s]
T_DOWN  = 20.0    # descend [s]
V_MAX   = 15.0    # cruise [m/s]
A_MAX   = 5.6     # max horiz accel [m/s^2]
R_EARTH = 6_371_000  # [m]

# -- false alarm --
FA_MEAN = 0.10
FA_STD  = 0.02

# -- sweep --
K_RANGE = range(1, 16)
TARGET_TIME = 90.0  # 1.5 min

# ═══════════════ HELPERS ═══════════════

def haversine(lon1, lat1, lon2, lat2):
    lo1, la1, lo2, la2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lo2 - lo1, la2 - la1
    a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
    return R_EARTH * 2 * np.arcsin(np.sqrt(a))

def drone_time(dist_m):
    d_accel = V_MAX**2 / (2 * A_MAX)
    if dist_m <= 0:
        return T_UP + T_DOWN
    if dist_m < 2 * d_accel:
        t_h = 2 * np.sqrt(dist_m / A_MAX)
    else:
        t_h = 2 * (V_MAX / A_MAX) + (dist_m - 2 * d_accel) / V_MAX
    return T_UP + t_h + T_DOWN

drone_time_vec = np.vectorize(drone_time)

def weighted_kmeans(lons, lats, weights, k, n_init=30):
    lon0, lat0 = lons.mean(), lats.mean()
    mx = haversine(lon0, lat0, lons, lat0) * np.sign(lons - lon0)
    my = haversine(lon0, lat0, lon0, lats) * np.sign(lats - lat0)
    coords = np.column_stack([mx, my])
    w_int = np.round(weights / weights.min()).astype(int)
    w_int = np.clip(w_int, 1, 10)
    expanded = np.repeat(coords, w_int, axis=0)
    km = KMeans(n_clusters=k, n_init=n_init, random_state=42)
    km.fit(expanded)
    centres_m = km.cluster_centers_
    c_lons = lon0 + centres_m[:, 0] / (R_EARTH * np.cos(np.radians(lat0))) * (180/np.pi)
    c_lats = lat0 + centres_m[:, 1] / R_EARTH * (180/np.pi)
    return c_lons, c_lats

def avg_response(lons, lats, weights, c_lons, c_lats):
    dists = np.zeros((len(lons), len(c_lons)))
    for j in range(len(c_lons)):
        dists[:, j] = haversine(lons, lats, c_lons[j], c_lats[j])
    nearest = dists.min(axis=1)
    times = drone_time_vec(nearest)
    return np.average(times, weights=weights)

def p95_response(lons, lats, c_lons, c_lats):
    dists = np.zeros((len(lons), len(c_lons)))
    for j in range(len(c_lons)):
        dists[:, j] = haversine(lons, lats, c_lons[j], c_lats[j])
    nearest = dists.min(axis=1)
    times = drone_time_vec(nearest)
    return np.percentile(times, 95)

def avg_distance(lons, lats, weights, c_lons, c_lats):
    """Weighted average straight-line distance from nearest station [m]."""
    dists = np.zeros((len(lons), len(c_lons)))
    for j in range(len(c_lons)):
        dists[:, j] = haversine(lons, lats, c_lons[j], c_lats[j])
    nearest = dists.min(axis=1)
    return np.average(nearest, weights=weights)

# ═══════════════ PLOTTING ═══════════════

CMAP_HEAT = LinearSegmentedColormap.from_list(
    "crime", ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#ff6b6b", "#ffe66d"])

def plot_map(ax, lons, lats, c_lons, c_lats, k, title=None, voronoi=True):
    ax.hexbin(lons, lats, gridsize=60, cmap=CMAP_HEAT, mincnt=1, alpha=0.85, zorder=1)
    if voronoi and k > 1:
        xg = np.linspace(lons.min()-0.002, lons.max()+0.002, 200)
        yg = np.linspace(lats.min()-0.001, lats.max()+0.001, 200)
        XX, YY = np.meshgrid(xg, yg)
        dists = np.zeros((XX.size, k))
        for j in range(k):
            dists[:, j] = haversine(XX.ravel(), YY.ravel(), c_lons[j], c_lats[j])
        labels = dists.argmin(axis=1).reshape(XX.shape)
        ax.contour(XX, YY, labels, levels=np.arange(k)+0.5,
                   colors='white', linewidths=0.6, alpha=0.45, zorder=2)
    ax.scatter(c_lons, c_lats, marker='*', s=350, c='#00ff88',
               edgecolors='white', linewidths=1.2, zorder=5)
    for i, (sx, sy) in enumerate(zip(c_lons, c_lats)):
        ax.annotate(f"S{i+1}", (sx, sy), fontsize=7, fontweight='bold',
                    color='white', ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points', zorder=6)
    ax.set_facecolor('#0d1117')
    ax.set_xlabel("Longitude", fontsize=8, color='#aaa')
    ax.set_ylabel("Latitude", fontsize=8, color='#aaa')
    ax.tick_params(colors='#888', labelsize=7)
    for s in ax.spines.values(): s.set_color('#333')
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', color='white', pad=8)

# ═══════════════ MAIN ═══════════════

print("Reading data ...")
df = pd.read_excel(INPUT)
lons = df["Longitude"].values
lats = df["Latitude"].values
df["False Alarm Rate"] = np.clip(np.random.normal(FA_MEAN, FA_STD, len(df)), 0, 1)
weights = 1.0 - df["False Alarm Rate"].values

print(f"  {len(df):,} incidents  |  FA mean={df['False Alarm Rate'].mean():.3f}")

# ── sweep ──
print("\nOptimising ...")
results = []
for k in K_RANGE:
    c_lo, c_la = weighted_kmeans(lons, lats, weights, k)
    t_avg = avg_response(lons, lats, weights, c_lo, c_la)
    t_95  = p95_response(lons, lats, c_lo, c_la)
    d_avg = avg_distance(lons, lats, weights, c_lo, c_la)
    results.append((k, t_avg, t_95, d_avg, c_lo, c_la))
    print(f"  k={k:2d}  avg={t_avg/60:.2f} min   P95={t_95/60:.2f} min   dist={d_avg:.0f} m")

# ── OUTPUT 1: grid of maps ──
print("\nPlotting station map grid ...")
fig, axes = plt.subplots(3, 4, figsize=(22, 15), facecolor='#0d1117')
fig.suptitle("Westminster Drone Station Placement — Crime-Density Weighted Optimisation",
             fontsize=16, fontweight='bold', color='white', y=0.98)
plot_ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
for idx, ax in enumerate(axes.flat):
    if idx >= len(plot_ks):
        ax.set_visible(False); continue
    kv = plot_ks[idx]
    r = [r for r in results if r[0]==kv][0]
    plot_map(ax, lons, lats, r[4], r[5], kv,
             title=f"{kv} Stn{'s' if kv>1 else ''}  —  avg {r[1]/60:.2f} min")
fig.tight_layout(rect=[0, 0, 1, 0.95])
p1 = os.path.join(OUTDIR, "station_maps_grid.png")
fig.savefig(p1, dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close(fig)
print(f"  -> {p1}")

# ── OUTPUT 2: response-time table + chart ──
print("Building response-time table ...")
tdata = [{"Stations": r[0],
          "Avg [s]": round(r[1],1), "Avg [min]": round(r[1]/60,2),
          "P95 [s]": round(r[2],1), "P95 [min]": round(r[2]/60,2),
          "Avg Dist [m]": round(r[3],1)} for r in results]
df_t = pd.DataFrame(tdata)
print(df_t.to_string(index=False))

fig2, (ax_c, ax_d, ax_tb) = plt.subplots(3, 1, figsize=(12, 13),
    gridspec_kw={"height_ratios": [2, 2, 1]}, facecolor='#0d1117')
ks   = [r[0] for r in results]
avgs = [r[1]/60 for r in results]
p95s = [r[2]/60 for r in results]
dsts = [r[3] for r in results]
ax_c.plot(ks, avgs, '-o', color='#00ff88', lw=2.5, ms=8, label='Avg Response', zorder=3)
ax_c.plot(ks, p95s, '-s', color='#ff6b6b', lw=2, ms=7, label='95th %ile', zorder=3)
ax_c.axhline(1.5, color='#ffe66d', ls='--', lw=1.5, alpha=0.8, label='1.5 min target')
ax_c.fill_between(ks, avgs, alpha=0.15, color='#00ff88')
ax_c.set_xlabel("Number of Stations", fontsize=11, color='#ccc')
ax_c.set_ylabel("Response Time [min]", fontsize=11, color='#ccc')
ax_c.set_title("Drone Response Time vs Number of Stations", fontsize=14,
                fontweight='bold', color='white')
ax_c.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
ax_c.set_facecolor('#0d1117'); ax_c.tick_params(colors='#aaa'); ax_c.set_xticks(ks)
ax_c.grid(axis='y', color='#222', alpha=0.6)
for s in ax_c.spines.values(): s.set_color('#333')

# distance subplot
ax_d.plot(ks, dsts, '-D', color='#bb86fc', lw=2.5, ms=8, label='Avg Distance', zorder=3)
ax_d.fill_between(ks, dsts, alpha=0.15, color='#bb86fc')
ax_d.set_xlabel("Number of Stations", fontsize=11, color='#ccc')
ax_d.set_ylabel("Avg Distance [m]", fontsize=11, color='#ccc')
ax_d.set_title("Avg Drone Travel Distance per Incident vs Number of Stations", fontsize=14,
                fontweight='bold', color='white')
ax_d.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
ax_d.set_facecolor('#0d1117'); ax_d.tick_params(colors='#aaa'); ax_d.set_xticks(ks)
ax_d.grid(axis='y', color='#222', alpha=0.6)
for s in ax_d.spines.values(): s.set_color('#333')

ax_tb.axis('off')
cols = ["Stations", "Avg [s]", "Avg [min]", "P95 [s]", "P95 [min]", "Avg Dist [m]"]
cells = [[str(r["Stations"]), f'{r["Avg [s]"]:.1f}', f'{r["Avg [min]"]:.2f}',
          f'{r["P95 [s]"]:.1f}', f'{r["P95 [min]"]:.2f}',
          f'{r["Avg Dist [m]"]:.1f}'] for r in tdata]
tbl = ax_tb.table(cellText=cells, colLabels=cols, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.4)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor('#444')
    if row == 0:
        cell.set_facecolor('#1a1a2e')
        cell.set_text_props(color='#00ff88', fontweight='bold')
    else:
        cell.set_facecolor('#0d1117')
        cell.set_text_props(color='white')
fig2.tight_layout()
p2 = os.path.join(OUTDIR, "response_time_table_chart.png")
fig2.savefig(p2, dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.close(fig2)
print(f"  -> {p2}")

# ── OUTPUT 3: ~1.5 min map ──
print("Finding config closest to 1.5 min ...")
target_r = None
for r in results:
    if r[1] <= TARGET_TIME:
        target_r = r; break
if target_r is None:
    target_r = min(results, key=lambda r: abs(r[1] - TARGET_TIME))
k_t, t_t, t95_t, d_t, clo_t, cla_t = target_r
print(f"  Selected k={k_t} (avg={t_t/60:.2f} min, dist={d_t:.0f} m)")

fig3, ax3 = plt.subplots(figsize=(14, 10), facecolor='#0d1117')
plot_map(ax3, lons, lats, clo_t, cla_t, k_t, voronoi=True)

# coverage ellipses for 1.5-min flight range
t_h_budget = TARGET_TIME - T_UP - T_DOWN
d_accel = V_MAX**2 / (2 * A_MAX)
t_accel_full = V_MAX / A_MAX
if t_h_budget > 2 * t_accel_full:
    max_range = 2 * d_accel + V_MAX * (t_h_budget - 2 * t_accel_full)
else:
    max_range = A_MAX * (t_h_budget / 2)**2
rdlon = max_range / (R_EARTH * np.cos(np.radians(lats.mean()))) * (180/np.pi)
rdlat = max_range / R_EARTH * (180/np.pi)
for sx, sy in zip(clo_t, cla_t):
    ax3.add_patch(mpatches.Ellipse((sx, sy), 2*rdlon, 2*rdlat,
        fill=False, edgecolor='#00ff88', lw=1.2, ls='--', alpha=0.6, zorder=4))

ax3.set_title(
    f"Optimal Layout for ~1.5 min Response:  {k_t} Stations  "
    f"(avg {t_t/60:.2f} min  |  P95 {t95_t/60:.2f} min)",
    fontsize=13, fontweight='bold', color='white', pad=12)
leg = [mpatches.Patch(facecolor='#e94560', alpha=0.7, label='Crime Density'),
       plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='#00ff88',
                  markersize=15, label='Drone Station', linestyle='None'),
       mpatches.Patch(fill=False, edgecolor='#00ff88', ls='--',
                      label=f'1.5-min range ({max_range:.0f} m)')]
ax3.legend(handles=leg, loc='upper left', fontsize=10,
           facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
fig3.tight_layout()
p3 = os.path.join(OUTDIR, "optimal_1_5min_stations.png")
fig3.savefig(p3, dpi=200, bbox_inches='tight', facecolor='#0d1117')
plt.close(fig3)
print(f"  -> {p3}")

# ── save enriched data ──
px = os.path.join(OUTDIR, "westminster_burglaries_with_FA.xlsx")
df.to_excel(px, index=False)
pc = os.path.join(OUTDIR, "response_time_table.csv")
df_t.to_csv(pc, index=False)
print(f"\nAll outputs saved to {OUTDIR}")
