import json
import numpy as np
from pathlib import Path

import geopandas as gpd
import rasterio
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "delhi_road_graph.json"
OUTPUT = ROOT / "data" / "processed" / "delhi_road_graph_compact.npz"

DEM_PATH = ROOT / "data" / "raw" / "dem" / "delhi_aw3d30.tif"
DRAINS_DIR = ROOT / "data" / "raw" / "drainage"


print("Loading original graph...")

with open(INPUT, "r", encoding="utf-8") as f:
    graph = json.load(f)

nodes = graph["nodes"]
edges = graph["edges"]

print(f"Nodes: {len(nodes):,}")
print(f"Edges: {len(edges):,}")


# ---------------------------------------------------------
# Map original OSM node IDs -> compact integer indices
# ---------------------------------------------------------

id_to_index = {
    int(node["id"]): i
    for i, node in enumerate(nodes)
}


# ---------------------------------------------------------
# Node arrays
# ---------------------------------------------------------

node_ids = np.array(
    [int(node["id"]) for node in nodes],
    dtype=np.int64,
)

lat = np.array(
    [float(node["lat"]) for node in nodes],
    dtype=np.float32,
)

lon = np.array(
    [float(node["lon"]) for node in nodes],
    dtype=np.float32,
)


# ---------------------------------------------------------
# Edge arrays
# ---------------------------------------------------------

sources = np.empty(len(edges), dtype=np.int32)
targets = np.empty(len(edges), dtype=np.int32)
distances = np.empty(len(edges), dtype=np.float32)


for i, edge in enumerate(edges):
    sources[i] = id_to_index[int(edge["source"])]
    targets[i] = id_to_index[int(edge["target"])]
    distances[i] = float(edge["distance_m"])


# ---------------------------------------------------------
# Sort edges by source node
# ---------------------------------------------------------

order = np.argsort(sources, kind="stable")

sources = sources[order]
targets = targets[order]
distances = distances[order]


# ---------------------------------------------------------
# Build CSR-style adjacency offsets
# ---------------------------------------------------------

node_count = len(nodes)

offsets = np.zeros(
    node_count + 1,
    dtype=np.int32,
)

np.add.at(
    offsets,
    sources + 1,
    1,
)

offsets = np.cumsum(
    offsets,
    dtype=np.int32,
)


# =========================================================
# PRECOMPUTE SPATIAL FLOOD-RISK MULTIPLIERS
# =========================================================

print()
print("Precomputing spatial flood-risk multipliers...")

node_coords = np.column_stack(
    (lat, lon)
).astype(
    np.float32,
    copy=False,
)

node_risk_mult = np.ones(
    node_count,
    dtype=np.float32,
)


# ---------------------------------------------------------
# DEM elevation
# ---------------------------------------------------------

if not DEM_PATH.exists():
    raise FileNotFoundError(
        f"DEM not found: {DEM_PATH}"
    )

print("Loading DEM...")

with rasterio.open(DEM_PATH) as dem_src:

    dem_data = dem_src.read(1)

    inv_transform = ~dem_src.transform

    lons = node_coords[:, 1]
    lats = node_coords[:, 0]

    cols, rows = inv_transform * (
        lons,
        lats,
    )

    rows = np.clip(
        rows.astype(np.int32),
        0,
        dem_data.shape[0] - 1,
    )

    cols = np.clip(
        cols.astype(np.int32),
        0,
        dem_data.shape[1] - 1,
    )

    elevations = dem_data[
        rows,
        cols,
    ]

    del dem_data


# ---------------------------------------------------------
# Drainage points
# ---------------------------------------------------------

drn_files = [
    DRAINS_DIR / "delhi_drains_mpd1976_full.geojson",
    DRAINS_DIR / "delhi_untraceable_drains_mpd1976-1.geojson",
]

drain_pts = []

print("Loading drainage data...")

for path in drn_files:

    if not path.exists():
        continue

    try:
        gdf = gpd.read_file(path)

        for geom in gdf.geometry:

            if geom is None or geom.is_empty:
                continue

            if geom.geom_type == "Point":

                drain_pts.append(
                    (geom.y, geom.x)
                )

            elif geom.geom_type in (
                "LineString",
                "LinearRing",
            ):

                drain_pts.extend(
                    (pt[1], pt[0])
                    for pt in geom.coords
                )

            elif geom.geom_type == "MultiLineString":

                for line in geom.geoms:

                    drain_pts.extend(
                        (pt[1], pt[0])
                        for pt in line.coords
                    )

        del gdf

    except Exception as exc:

        print(
            f"Warning: could not read {path.name}: {exc}"
        )


# ---------------------------------------------------------
# Calculate distance to nearest drain
# ---------------------------------------------------------

if drain_pts:

    drain_pts_arr = np.asarray(
        drain_pts,
        dtype=np.float32,
    )

    print(
        f"Drainage points: {len(drain_pts_arr):,}"
    )

    drain_tree = cKDTree(
        drain_pts_arr
    )

    dist_deg, _ = drain_tree.query(
        node_coords
    )

    dist_m = (
        dist_deg * 111_000.0
    ).astype(
        np.float32
    )

    del drain_pts_arr
    del drain_tree

else:

    print(
        "Warning: no drainage points found."
    )

    dist_m = np.full(
        node_count,
        1000.0,
        dtype=np.float32,
    )


# ---------------------------------------------------------
# Same spatial proxy thresholds as routing.py
# ---------------------------------------------------------

high_mask = (
    (elevations < 210.0)
    & (dist_m < 50.0)
)

medium_mask = (
    (elevations < 215.0)
    & (dist_m < 200.0)
    & (~high_mask)
)

node_risk_mult[
    high_mask
] = 5.0

node_risk_mult[
    medium_mask
] = 2.0


print(
    f"High-risk nodes: {np.sum(high_mask):,}"
)

print(
    f"Medium-risk nodes: {np.sum(medium_mask):,}"
)

print(
    f"Low-risk nodes: "
    f"{np.sum(node_risk_mult == 1.0):,}"
)


del elevations
del dist_m
del high_mask
del medium_mask
del node_coords


# =========================================================
# SAVE COMPACT DEPLOYMENT GRAPH
# =========================================================

print()
print("Saving compact graph with precomputed risk...")

np.savez(
    OUTPUT,

    # Original node identity
    node_ids=node_ids,

    # Coordinates
    lat=lat,
    lon=lon,

    # CSR adjacency
    offsets=offsets,
    targets=targets,
    distances=distances,

    # Precomputed spatial flood-risk multiplier
    node_risk_mult=node_risk_mult,
)


print()
print("Done.")

print(
    f"Output: {OUTPUT}"
)

print(
    f"Size: "
    f"{OUTPUT.stat().st_size / (1024 * 1024):.2f} MB"
)

print(
    f"Risk array size: "
    f"{node_risk_mult.nbytes / (1024 * 1024):.2f} MB"
)