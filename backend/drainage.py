"""
AquaG Drainage Network Backend Engine (Stage 12 Phase 3A)

Loads MPD-1976 documented drainage reference locations, filters by viewport bounding box,
and returns GeoJSON FeatureCollection preserving source Point geometries and attributes.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from shared_resources import find_project_file
from waterlogging import clamp_and_validate_bbox, DELHI_OPERATING_BOUNDS

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
            with open(DRAIN_FULL_PATH, "r", encoding="utf-8") as f:
                d_data = json.load(f)
            for feat in d_data.get("features", []):
                geom = feat.get("geometry")
                props = feat.get("properties", {})
                if not geom or geom.get("type") != "Point":
                    continue
                coords = geom.get("coordinates")
                if not coords or len(coords) < 2:
                    continue
                lon, lat = float(coords[0]), float(coords[1])
                seq_val = props.get("seq_no", 0)
                seq_no = int(seq_val) if str(seq_val).isdigit() else 0
                features.append({
                    "lon": lon,
                    "lat": lat,
                    "drain_name": str(props.get("drain_name", "MPD-1976 Drain")),
                    "basin": str(props.get("basin", "Delhi Basin")),
                    "seq_no": seq_no,
                    "status": str(props.get("status", "Existing / Remodeling")),
                    "source": str(props.get("source", "MPD-1976")),
                    "geometry_type": "Point"
                })
        except Exception:
            pass

    # 2. Load Untraceable Drains GeoJSON
    if UNTRACE_PATH.exists():
        try:
            with open(UNTRACE_PATH, "r", encoding="utf-8") as f:
                u_data = json.load(f)
            for feat in u_data.get("features", []):
                geom = feat.get("geometry")
                props = feat.get("properties", {})
                if not geom or geom.get("type") != "Point":
                    continue
                coords = geom.get("coordinates")
                if not coords or len(coords) < 2:
                    continue
                lon, lat = float(coords[0]), float(coords[1])
                seq_val = props.get("seq_no", 0)
                seq_no = int(seq_val) if str(seq_val).isdigit() else 0
                features.append({
                    "lon": lon,
                    "lat": lat,
                    "drain_name": str(props.get("drain_name", "Untraceable Drain")),
                    "basin": str(props.get("basin", "Delhi Basin")),
                    "seq_no": seq_no,
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

    # Bounding Box Filtering & Clamping
    clamped_bbox, is_valid = clamp_and_validate_bbox(bbox)
    if not is_valid or clamped_bbox is None:
        exec_ms = round((time.time() - t_start) * 1000.0, 1)
        return {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "total_matched": 0,
                "returned_features": 0,
                "max_features_cap": max_features,
                "bbox": bbox,
                "execution_ms": exec_ms,
                "geometry": "Point",
                "source": "MPD-1976",
                "status": "outside_operating_area",
            },
        }

    min_lon, min_lat, max_lon, max_lat = clamped_bbox
    filtered = [
        item for item in _DRAINAGE_CACHE
        if min_lon <= item["lon"] <= max_lon and min_lat <= item["lat"] <= max_lat
    ]

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
