"""
AquaG Drainage Network Backend Engine (Stage 12 Phase 3A)

Loads MPD-1976 documented drainage reference locations, filters by viewport bounding box,
and returns GeoJSON FeatureCollection preserving source Point geometries and attributes.
"""

from __future__ import annotations
import json
import time
import geopandas as gpd
from pathlib import Path
from typing import Dict, Any, List, Optional

from waterlogging import find_project_file

# ---------------------------------------------------------------------------
# Path Resolutions
# ---------------------------------------------------------------------------
DRAIN_FULL_PATH = find_project_file("existing code/data/raw/drainage/delhi_drains_mpd1976_full.geojson")
UNTRACE_PATH = find_project_file("existing code/data/raw/drainage/delhi_untraceable_drains_mpd1976-1.geojson")

_DRAINAGE_CACHE: Optional[List[Dict[str, Any]]] = None


def _init_drainage_cache() -> None:
    """Lazy load documented MPD-1976 drainage features once into memory."""
    global _DRAINAGE_CACHE
    if _DRAINAGE_CACHE is not None:
        return

    features = []

    # 1. Load Main Drainage GeoJSON
    if DRAIN_FULL_PATH.exists():
        try:
            gdf_main = gpd.read_file(DRAIN_FULL_PATH)
            for _, row in gdf_main.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                
                lon, lat = float(geom.x), float(geom.y)
                features.append({
                    "lon": lon,
                    "lat": lat,
                    "drain_name": str(row.get("drain_name", "MPD-1976 Drain")),
                    "basin": str(row.get("basin", "Delhi Basin")),
                    "seq_no": int(row.get("seq_no", 0)) if str(row.get("seq_no", "")).isdigit() else 0,
                    "status": str(row.get("status", "Existing / Remodeling")),
                    "source": str(row.get("source", "MPD-1976")),
                    "geometry_type": "Point"
                })
        except Exception:
            pass

    # 2. Load Untraceable Drains GeoJSON
    if UNTRACE_PATH.exists():
        try:
            gdf_untrace = gpd.read_file(UNTRACE_PATH)
            for _, row in gdf_untrace.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                lon, lat = float(geom.x), float(geom.y)
                features.append({
                    "lon": lon,
                    "lat": lat,
                    "drain_name": str(row.get("drain_name", "Untraceable Drain")),
                    "basin": str(row.get("basin", "Delhi Basin")),
                    "seq_no": int(row.get("seq_no", 0)) if str(row.get("seq_no", "")).isdigit() else 0,
                    "status": "Untraceable / Encroached",
                    "source": "MPD-1976 Untraceable",
                    "geometry_type": "Point"
                })
        except Exception:
            pass

    _DRAINAGE_CACHE = features


def get_drainage_geojson(
    bbox: Optional[List[float]] = None,
    max_features: int = 1000
) -> Dict[str, Any]:
    """
    Returns viewport-filtered GeoJSON FeatureCollection of drainage network reference points.
    """
    _init_drainage_cache()
    t_start = time.time()

    if _DRAINAGE_CACHE is None or len(_DRAINAGE_CACHE) == 0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {"total": 0, "returned": 0, "source": "MPD-1976"}
        }

    # Bounding Box Filtering
    filtered = []
    if bbox and len(bbox) == 4:
        min_lon, min_lat, max_lon, max_lat = bbox
        for item in _DRAINAGE_CACHE:
            if min_lon <= item["lon"] <= max_lon and min_lat <= item["lat"] <= max_lat:
                filtered.append(item)
    else:
        filtered = _DRAINAGE_CACHE

    total_matched = len(filtered)
    filtered_capped = filtered[:max_features]

    features = []
    for item in filtered_capped:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [item["lon"], item["lat"]]
            },
            "properties": {
                "drain_name": item["drain_name"],
                "basin": item["basin"],
                "seq_no": item["seq_no"],
                "status": item["status"],
                "source": item["source"],
                "geometry_type": item["geometry_type"]
            }
        }
        features.append(feature)

    exec_ms = round((time.time() - t_start) * 1000.0, 1)

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_matched": total_matched,
            "returned_features": len(features),
            "max_features_cap": max_features,
            "bbox": bbox,
            "execution_ms": exec_ms,
            "geometry": "Point",
            "source": "MPD-1976"
        }
    }
