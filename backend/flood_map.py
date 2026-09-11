"""
AquaG Flood Map GIS Utilities

Stage 6:
- DEM elevation queries
- Nearest drain queries
- Spatial flood proxy information

Deployment optimized:
- DEM is NOT fully loaded into RAM at startup.
- Drain GeoJSON files are NOT loaded at startup.
- Population GeoJSON is NOT loaded at startup.
- Data is loaded only when the corresponding function is called.
"""

from pathlib import Path

import rasterio
from shapely.geometry import Point


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_ROOT = PROJECT_ROOT / "data" / "raw"

DEM_PATH = (
    DATA_RAW_ROOT
    / "dem"
    / "delhi_aw3d30.tif"
)

DRAINS_DIR = (
    DATA_RAW_ROOT
    / "drainage"
)

DRN_FILES = [
    DRAINS_DIR
    / "delhi_drains_mpd1976_full.geojson",

    DRAINS_DIR
    / "delhi_untraceable_drains_mpd1976-1.geojson",
]

POP_PATH = (
    DATA_RAW_ROOT
    / "population"
    / "delhi_districts_population_2011-3.geojson"
)


# ============================================================
# LAZY-LOADED DATA
# ============================================================

# These variables intentionally start as None.
# They will be loaded only when required.

DEMDataset = None
DEM_CRS = None
DEM_BOUNDS = None

DRAINS_GDF = None

POP_GDF = None


# ============================================================
# DEM LOADER
# ============================================================

def _load_dem():
    """
    Open the DEM only when a DEM query is required.

    The complete raster is NOT read into RAM.
    """

    global DEMDataset
    global DEM_CRS
    global DEM_BOUNDS

    if DEMDataset is not None:
        return DEMDataset

    if not DEM_PATH.exists():
        raise FileNotFoundError(
            f"DEM file not found: {DEM_PATH}"
        )

    print("Loading DEM dataset lazily...")

    DEMDataset = rasterio.open(
        DEM_PATH
    )

    DEM_CRS = DEMDataset.crs
    DEM_BOUNDS = DEMDataset.bounds

    return DEMDataset


# ============================================================
# DRAIN LOADER
# ============================================================

def _load_drains():
    """
    Load drainage GeoJSON files only when a drain query
    is actually requested.
    """

    global DRAINS_GDF

    if DRAINS_GDF is not None:
        return DRAINS_GDF

    import geopandas as gpd

    print("Loading drainage data lazily...")

    frames = []

    for path in DRN_FILES:

        if not path.exists():
            continue

        try:
            gdf = gpd.read_file(
                path
            )

            if gdf.empty:
                continue

            if gdf.crs is None:

                gdf = gdf.set_crs(
                    epsg=4326
                )

            elif gdf.crs.to_epsg() != 4326:

                gdf = gdf.to_crs(
                    epsg=4326
                )

            frames.append(gdf)

        except Exception as exc:

            print(
                f"Warning: could not load drain file "
                f"{path.name}: {exc}"
            )

    if not frames:

        raise ValueError(
            "No drain features loaded"
        )

    # Concatenate GeoDataFrames without pandas import.
    DRAINS_GDF = gpd.GeoDataFrame(
        gpd.pd.concat(
            frames,
            ignore_index=True,
        ),
        crs="EPSG:4326",
    )

    return DRAINS_GDF


# ============================================================
# POPULATION LOADER
# ============================================================

def _load_population():
    """
    Load population data only when needed.
    """

    global POP_GDF

    if POP_GDF is not None:
        return POP_GDF

    if not POP_PATH.exists():
        return None

    import geopandas as gpd

    print("Loading population data lazily...")

    try:

        POP_GDF = gpd.read_file(
            POP_PATH
        )

        if POP_GDF.crs is None:

            POP_GDF = POP_GDF.set_crs(
                epsg=4326
            )

        elif POP_GDF.crs.to_epsg() != 4326:

            POP_GDF = POP_GDF.to_crs(
                epsg=4326
            )

        return POP_GDF

    except Exception as exc:

        print(
            f"Warning: could not load population data: {exc}"
        )

        POP_GDF = None

        return None


# ============================================================
# NEAREST DRAIN
# ============================================================

def nearest_drain(
    lat: float,
    lon: float,
) -> dict:
    """
    Return the nearest drain feature and distance in metres.

    Drain data is loaded lazily.
    """

    drains = _load_drains()

    if drains.empty:
        raise ValueError(
            "No drain features available."
        )

    point = Point(
        lon,
        lat,
    )

    # Build/use GeoPandas spatial index.
    nearest_result = drains.sindex.nearest(
        point
    )

    # GeoPandas versions may return a 2 x N array.
    if hasattr(
        nearest_result,
        "shape",
    ) and len(nearest_result.shape) > 1:

        nearest_idx = int(
            nearest_result[1][0]
        )

    else:

        nearest_idx = int(
            nearest_result[0]
        )

    nearest_geom = (
        drains.geometry.iloc[
            nearest_idx
        ]
    )

    if nearest_geom is None:
        raise ValueError(
            "Nearest drain geometry is unavailable."
        )

    # Coordinates are EPSG:4326.
    # This is the same spatial proxy used previously.
    deg_distance = (
        nearest_geom.distance(
            point
        )
    )

    meters = (
        float(deg_distance)
        * 111_000.0
    )

    return {
        "drain_id": nearest_idx,
        "distance_m": meters,
        "geometry": nearest_geom.wkt,
    }


# ============================================================
# DEM ELEVATION
# ============================================================

def dem_elevation(
    lat: float,
    lon: float,
) -> float:
    """
    Return DEM elevation in metres.

    IMPORTANT:
    The old implementation loaded the entire DEM into RAM.

    This version reads only a tiny window around the requested
    coordinate, greatly reducing memory usage.
    """

    dem = _load_dem()

    query_lon = lon
    query_lat = lat

    # Transform coordinates if DEM CRS is not EPSG:4326.
    if (
        dem.crs is not None
        and dem.crs.to_epsg() != 4326
    ):

        from pyproj import Transformer

        transformer = Transformer.from_crs(
            "EPSG:4326",
            dem.crs,
            always_xy=True,
        )

        query_lon, query_lat = (
            transformer.transform(
                lon,
                lat,
            )
        )

    try:

        row, col = dem.index(
            query_lon,
            query_lat,
        )

    except Exception as exc:

        raise ValueError(
            "Could not convert coordinates "
            "to DEM pixel."
        ) from exc

    # Check bounds before reading.
    if (
        row < 0
        or row >= dem.height
        or col < 0
        or col >= dem.width
    ):

        raise ValueError(
            "Coordinates out of DEM bounds"
        )

    # Read ONLY the requested pixel.
    window = rasterio.windows.Window(
        col_off=col,
        row_off=row,
        width=1,
        height=1,
    )

    value = dem.read(
        1,
        window=window,
    )[0, 0]

    return float(value)


# ============================================================
# FLOOD INFORMATION
# ============================================================

def get_flood_info(
    lat: float,
    lon: float,
) -> dict:
    """
    Aggregate flood information for a point.

    Includes:
    - DEM elevation
    - nearest documented MPD-1976 drain
    - spatial proxy basis
    """

    try:

        elev = dem_elevation(
            lat,
            lon,
        )

    except Exception:

        elev = None

    try:

        drain = nearest_drain(
            lat,
            lon,
        )

        drain_dist = (
            drain.get(
                "distance_m"
            )
            if drain
            else None
        )

    except Exception:

        drain = None
        drain_dist = None

    return {
        "latitude": lat,
        "longitude": lon,

        "elevation": elev,

        "elevation_m": (
            round(elev, 2)
            if elev is not None
            else None
        ),

        "nearest_drain": drain,

        "nearest_drain_distance_m": (
            round(
                drain_dist,
                2,
            )
            if drain_dist is not None
            else None
        ),

        "risk_basis": "spatial_proxy",
    }