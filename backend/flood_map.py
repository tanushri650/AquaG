"""
AquaG Flood Map GIS Utilities
Stage 6: GIS layer queries including DEM elevation and nearest drain distance.
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import Point

# Base data directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_ROOT = PROJECT_ROOT / "data" / "raw"

# Load DEM dataset into memory once at module import
DEM_PATH = DATA_RAW_ROOT / "dem" / "delhi_aw3d30.tif"
if not DEM_PATH.exists():
    raise FileNotFoundError(f"DEM file not found: {DEM_PATH}")
DEMDataset = rasterio.open(DEM_PATH)
DEM_CRS = DEMDataset.crs
DEM_BOUNDS = DEMDataset.bounds
DEM_DATA = DEMDataset.read(1)

# Load drains GeoJSON
DRAINS_DIR = DATA_RAW_ROOT / "drainage"
DRN_FILES = [
    DRAINS_DIR / "delhi_drains_mpd1976_full.geojson",
    DRAINS_DIR / "delhi_untraceable_drains_mpd1976-1.geojson",
]
DRAINS_GDF = gpd.GeoDataFrame(
    pd.concat(
        [gpd.read_file(p) for p in DRN_FILES if p.exists()],
        ignore_index=True,
    )
)
if DRAINS_GDF.empty:
    raise ValueError("No drain features loaded")

if DRAINS_GDF.crs is None:
    DRAINS_GDF.set_crs(epsg=4326, inplace=True)
else:
    DRAINS_GDF = DRAINS_GDF.to_crs(epsg=4326)

# Load population GeoJSON
POP_PATH = DATA_RAW_ROOT / "population" / "delhi_districts_population_2011-3.geojson"
POP_GDF = None
if POP_PATH.exists():
    POP_GDF = gpd.read_file(POP_PATH)
    if POP_GDF.crs is None:
        POP_GDF.set_crs(epsg=4326, inplace=True)
    else:
        POP_GDF = POP_GDF.to_crs(epsg=4326)


def nearest_drain(lat: float, lon: float) -> dict:
    """Return the nearest drain feature and distance (meters) using spatial index."""
    point = Point(lon, lat)
    nearest_idx = DRAINS_GDF.sindex.nearest(point)[1][0]
    nearest_geom = DRAINS_GDF.geometry.iloc[nearest_idx]
    deg_distance = nearest_geom.distance(point)
    meters = deg_distance * 111_000
    return {
        "drain_id": int(nearest_idx),
        "distance_m": float(meters),
        "geometry": nearest_geom.wkt,
    }


def dem_elevation(lat: float, lon: float) -> float:
    """Return DEM elevation (meters) at given lat/lon using cached DEM memory array."""
    if DEMDataset.crs.to_epsg() != 4326:
        from pyproj import Transformer
        transformer = Transformer.from_crs("epsg:4326", DEMDataset.crs, always_xy=True)
        lon, lat = transformer.transform(lon, lat)
    row, col = DEMDataset.index(lon, lat)
    try:
        value = DEM_DATA[row, col]
    except IndexError:
        raise ValueError("Coordinates out of DEM bounds")
    return float(value)


def get_flood_info(lat: float, lon: float) -> dict:
    """Aggregate flood info for a point: elevation and nearest documented MPD-1976 drain location."""
    try:
        elev = dem_elevation(lat, lon)
    except Exception:
        elev = None
    try:
        drain = nearest_drain(lat, lon)
        drain_dist = drain.get("distance_m") if drain else None
    except Exception:
        drain = None
        drain_dist = None

    return {
        "latitude": lat,
        "longitude": lon,
        "elevation": elev,
        "elevation_m": round(elev, 2) if elev is not None else None,
        "nearest_drain": drain,
        "nearest_drain_distance_m": round(drain_dist, 2) if drain_dist is not None else None,
        "risk_basis": "spatial_proxy",
    }
