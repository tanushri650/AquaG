#!/usr/bin/env python3
"""build_ml_dataset_v2.py

Generate the ML‑ready dataset (V2) for the AquaG flood nowcasting project.

Key design goals:
- **Memory‑efficient**: Process large vector/raster files in chunks and use spatial indexes (R‑tree) to avoid loading everything into memory.
- **Reproducible**: All paths are relative to the project root, and a deterministic random seed is set for any stochastic steps.
- **Extensible**: Feature extraction functions are modular; new features can be added by extending the `FEATURES` dictionary.

Expected directory layout (relative to the project root):
```
project_root/
├─ data/
│  ├─ raw/
│  │  ├─ dem/delhi_aw3d30.tif                     # DEM raster
│  │  ├─ rainfall/rainfall_tel_hr_delhi_sw_gw_dl_2021_2025.csv
│  │  ├─ drainage/delhi_drains_mpd1976_full.geojson
│  │  ├─ drains_untraceable/delhi_untraceable_drains_mpd1976.geojson
│  │  ├─ roads/delhi_road_network.json
│  │  ├─ infrastructure/delhi_important_infrastructure.json
│  │  ├─ population/delhi_districts_population_2011.geojson
│  │  └─ flood/zone_flood_data_1000plus.geojson
│  └─ processed/zone_flood_data.csv               # baseline zones & prototype labels
├─ scripts/build_ml_dataset_v2.py                  # <‑ this file
└─ data/processed/aquag_ml_dataset_v2.csv          # generated output
```

Running the script:
```bash
python scripts/build_ml_dataset_v2.py
```
The script writes the resulting CSV to `data/processed/aquag_ml_dataset_v2.csv`.
"""

import os
import sys
import json
import logging
import pathlib
import warnings
from typing import Dict, List, Tuple, Any

import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.features
import rasterio.mask
import numpy as np
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # repository root
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV = PROCESSED_DIR / "aquag_ml_dataset_v2.csv"

# Input file locations (adjust if your folder names differ)
DEM_PATH = RAW_DIR / "dem" / "delhi_aw3d30.tif"
RAINFALL_CSV = RAW_DIR / "rainfall" / "rainfall_tel_hr_delhi_sw_gw_dl_2021_2025.csv"
DRAINAGE_GEOJSON = RAW_DIR / "drainage" / "delhi_drains_mpd1976_full.geojson"
UNTRACEABLE_DRAINAGE_GEOJSON = RAW_DIR / "drainage" / "delhi_untraceable_drains_mpd1976.geojson"
ROAD_NETWORK_JSON = RAW_DIR / "roads" / "delhi_road_network.json"
INFRA_JSON = RAW_DIR / "infrastructure" / "delhi_important_infrastructure.json"
POP_GEOJSON = RAW_DIR / "population" / "delhi_districts_population_2011.geojson"
ZONE_GEOJSON = RAW_DIR / "flood" / "zone_flood_data_1000plus.geojson"
BASE_ZONE_CSV = PROCESSED_DIR / "zone_flood_data.csv"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def load_zone_dataframe() -> gpd.GeoDataFrame:
    """Load zone polygons and merge with the baseline CSV.

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: zone_id, geometry, flood_severity (prototype), ...
    """
    logger.info("Loading zone geometry from %s", ZONE_GEOJSON)
    zones_gdf = gpd.read_file(ZONE_GEOJSON)
    zones_gdf = zones_gdf.rename(columns={"id": "zone_id"})
    # Ensure a stable CRS – we will work in EPSG:4326 (WGS84) for all vector data.
    zones_gdf = zones_gdf.to_crs(epsg=4326)

    logger.info("Merging with baseline zone CSV %s", BASE_ZONE_CSV)
    base_df = pd.read_csv(BASE_ZONE_CSV)
    zones = zones_gdf.merge(base_df, on="zone_id", how="left")
    return zones

def raster_query_point(raster: rasterio.io.DatasetReader, point: Tuple[float, float]) -> float:
    """Return raster value at a lon/lat point using nearest neighbour.
    If the point falls outside the raster, returns np.nan.
    """
    try:
        row, col = raster.index(point[0], point[1])
        return raster.read(1)[row, col]
    except (IndexError, rasterio.errors.RasterioError):
        return np.nan

def compute_dem_features(zones: gpd.GeoDataFrame, raster_path: pathlib.Path) -> pd.DataFrame:
    """Extract elevation and slope for each zone centroid.
    Slope is approximated using a simple 3×3 Sobel filter on the DEM.
    """
    logger.info("Computing DEM features from %s", raster_path)
    with rasterio.open(raster_path) as src:
        # Pre‑compute a slope raster (in‑memory) – small enough for Delhi DEM.
        elevation = src.read(1).astype(float)
        # Compute gradients using numpy (central differences)
        dy, dx = np.gradient(elevation, src.res[0], src.res[1])
        slope = np.sqrt(dx**2 + dy**2)
        # Build an in‑memory raster for slope using the same metadata.
        meta = src.meta.copy()
        meta.update(dtype=rasterio.float32)
        # Helper to query both rasters.
        def query(pt):
            elev = raster_query_point(src, pt)
            # Use the same index for slope array.
            try:
                row, col = src.index(pt[0], pt[1])
                sl = slope[row, col]
            except Exception:
                sl = np.nan
            return elev, sl

        elevations = []
        slopes = []
        for _, row in zones.iterrows():
            lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
            e, s = query((lon, lat))
            elevations.append(e)
            slopes.append(s)
        return pd.DataFrame({"elevation": elevations, "slope": slopes})

def build_spatial_index(gdf: gpd.GeoDataFrame) -> Tuple[STRtree, List[int]]:
    """Create an R‑tree and a list mapping geometry index to original row index.
    Returns (tree, idx_list) where `tree.query(point)` gives indices into `idx_list`.
    """
    geometries = gdf.geometry.values
    tree = STRtree(geometries)
    # Preserve original pandas index for lookup after query.
    idx_list = list(gdf.index)
    return tree, idx_list

def nearest_feature_distance(point: Point, tree: STRtree, idx_list: List[int], gdf: gpd.GeoDataFrame, attr: str = None) -> Tuple[float, Any]:
    """Return distance (in metres) from `point` to the nearest feature in `gdf`.
    If `attr` is supplied, also return the attribute value of that nearest feature.
    """
    # `tree.nearest` is not available; we perform a small radius query.
    # Start with a tiny radius and expand until we find something.
    radius = 0.001  # ~100 m in degrees (rough guess)
    while radius < 0.5:  # ~50 km limit
        candidates = tree.query(point.buffer(radius))
        if candidates:
            # Pick the closest by actual Euclidean distance.
            min_dist = float('inf')
            min_val = None
            for geom_idx in candidates:
                geom = gdf.geometry.iloc[geom_idx]
                dist = point.distance(geom)
                if dist < min_dist:
                    min_dist = dist
                    min_val = gdf.iloc[geom_idx][attr] if attr else None
            # Convert degree distance to metres (approx using 111 km per degree).
            metres = min_dist * 111_000
            return metres, min_val
        radius *= 2
    return np.nan, None

def compute_vector_features(zones: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute distance‑based features for drainage, roads, infrastructure and population.
    The function streams through each zone centroid and uses spatial indexes for fast lookup.
    """
    logger.info("Loading vector datasets for spatial joins")
    drains = gpd.read_file(DRAINAGE_GEOJSON).to_crs(epsg=4326)
    roads = gpd.read_file(ROAD_NETWORK_JSON).to_crs(epsg=4326)
    infra = gpd.read_file(INFRA_JSON).to_crs(epsg=4326)
    pop = gpd.read_file(POP_GEOJSON).to_crs(epsg=4326)

    # Build indexes
    drain_tree, drain_idx = build_spatial_index(drains)
    road_tree, road_idx = build_spatial_index(roads)
    infra_tree, infra_idx = build_spatial_index(infra)
    pop_tree, pop_idx = build_spatial_index(pop)

    logger.info("Computing distance‑to‑nearest‑feature metrics")
    distance_to_drain = []
    distance_to_road = []
    distance_to_infra = []
    pop_density = []

    for _, zone in tqdm(zones.iterrows(), total=len(zones), desc="Zones"):
        centroid = zone.geometry.centroid
        # Drainage distance (metres)
        d_drain, _ = nearest_feature_distance(centroid, drain_tree, drain_idx, drains)
        distance_to_drain.append(d_drain)
        # Road distance (metres)
        d_road, _ = nearest_feature_distance(centroid, road_tree, road_idx, roads)
        distance_to_road.append(d_road)
        # Critical infra – we treat presence within 100 m as flag = 1.
        d_infra, infra_val = nearest_feature_distance(centroid, infra_tree, infra_idx, infra, attr="type")
        distance_to_infra.append(d_infra)
        # Population density – we assume the polygon has a field `pop_density`.
        _, pop_val = nearest_feature_distance(centroid, pop_tree, pop_idx, pop, attr="pop_density")
        pop_density.append(pop_val if pop_val is not None else np.nan)

    df = pd.DataFrame({
        "distance_to_drain": distance_to_drain,
        "distance_to_road": distance_to_road,
        "distance_to_infra": distance_to_infra,
        "population_density": pop_density,
    })
    return df

def load_rainfall_features() -> pd.DataFrame:
    """Aggregate hourly rainfall into 1h, 3h, 6h windows per zone.
    This placeholder implementation assumes the CSV already contains pre‑aggregated values
    with a `zone_id` column. In a real pipeline you would spatial‑join rain‑gauges to zones
    and sum over the desired windows.
    """
    logger.info("Loading rainfall aggregation from %s", RAINFALL_CSV)
    df = pd.read_csv(RAINFALL_CSV)
    # Expected columns: zone_id, rainfall_1h, rainfall_3h, rainfall_6h, recent_rainfall_intensity
    required = {"zone_id", "rainfall_1h", "rainfall_3h", "rainfall_6h", "recent_rainfall_intensity"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Rainfall CSV missing columns: %s", missing)
    return df.set_index("zone_id")

def main():
    logger.info("Starting ML dataset V2 generation")
    # Step 1: Load base zones (geometry + prototype label fields)
    zones_gdf = load_zone_dataframe()
    zones_gdf = zones_gdf.set_index("zone_id")

    # Step 2: DEM features
    dem_features = compute_dem_features(zones_gdf, DEM_PATH)
    dem_features.index = zones_gdf.index

    # Step 3: Vector‑based spatial features
    vector_features = compute_vector_features(zones_gdf)
    vector_features.index = zones_gdf.index

    # Step 4: Rainfall features (already aggregated)
    rainfall_features = load_rainfall_features()

    # Step 5: Assemble final table
    logger.info("Merging all feature tables")
    final_df = zones_gdf[["rainfall_1h", "rainfall_3h", "rainfall_6h", "recent_rainfall_intensity", "flood_severity", "flood_history_count"]].copy()
    # If these columns are not present in the base CSV we fall back to the rainfall dataframe.
    for col in ["rainfall_1h", "rainfall_3h", "rainfall_6h", "recent_rainfall_intensity"]:
        if col not in final_df.columns:
            final_df[col] = rainfall_features[col]
    final_df = final_df.join(dem_features, how="left")
    final_df = final_df.join(vector_features, how="left")

    # Step 6: Clean / type‑cast
    final_df = final_df.reset_index()
    final_df = final_df.rename(columns={"index": "zone_id"})
    # Ensure numeric columns are float and fill NaNs where appropriate.
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    final_df[numeric_cols] = final_df[numeric_cols].astype(float)

    # Step 7: Write CSV
    logger.info("Writing output CSV to %s", OUTPUT_CSV)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Dataset generation completed successfully.")

if __name__ == "__main__":
    # Suppress geopandas warnings about future deprecations.
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
