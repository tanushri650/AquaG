"""
AquaG Critical Infrastructure Backend Engine (Stage 12 Phase 3A)

Loads OpenStreetMap-derived Delhi critical infrastructure dataset lazily,
filters by viewport bounding box, classifies features into standard categories,
and returns viewport-capped GeoJSON FeatureCollection.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from waterlogging import find_project_file, clamp_and_validate_bbox, DELHI_OPERATING_BOUNDS

# ---------------------------------------------------------------------------
# Path Resolutions & Constants
# ---------------------------------------------------------------------------
INFRA_PATH = find_project_file("existing code/data/raw/infrastructure/delhi_important_infrastructure.json")

_INFRA_CACHE: Optional[List[Dict[str, Any]]] = None


def _classify_infra_feature(tags: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Classify OSM tags into standard categories:
    hospital, police, emergency, railway, metro, school, college, other.
    Returns (category_name, is_critical_boolean).
    """
    amenity = str(tags.get("amenity", "")).lower()
    railway = str(tags.get("railway", "")).lower()
    subway = str(tags.get("subway", "")).lower()
    network = str(tags.get("network", "")).lower()
    healthcare = str(tags.get("healthcare", "")).lower()
    emergency = str(tags.get("emergency", "")).lower()
    building = str(tags.get("building", "")).lower()

    # 1. Hospital / Healthcare
    if amenity == "hospital" or healthcare == "hospital" or "hospital" in amenity or "hospital" in healthcare:
        return "hospital", True
    if amenity in ("clinic", "doctors", "pharmacy") or healthcare in ("clinic", "centre"):
        return "hospital", False

    # 2. Police / Fire / Emergency
    if amenity == "police" or building == "police":
        return "police", True
    if emergency != "" or amenity == "fire_station" or "fire" in amenity:
        return "emergency", True

    # 3. Metro / Subway Transit
    if "metro" in network or subway == "yes" or railway in ("subway", "subway_entrance"):
        return "metro", True

    # 4. Railway Infrastructure
    if railway in ("station", "rail", "stop", "level_crossing", "halt", "platform", "junction"):
        return "railway", True

    # 5. School
    if amenity == "school" or building == "school":
        return "school", False

    # 6. College / University
    if amenity in ("college", "university") or building in ("college", "university"):
        return "college", False

    return "other", False


def _init_infrastructure_cache() -> None:
    """Lazy load OSM infrastructure dataset once into memory."""
    global _INFRA_CACHE
    if _INFRA_CACHE is not None:
        return

    if not INFRA_PATH.exists():
        _INFRA_CACHE = []
        return

    try:
        with open(INFRA_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception:
        _INFRA_CACHE = []
        return

    elements = raw_data.get("elements", [])
    parsed_features = []

    for el in elements:
        tags = el.get("tags", {})
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")

        if lat is None or lon is None:
            continue

        lat = float(lat)
        lon = float(lon)
        if not (28.0 <= lat <= 29.2 and 76.5 <= lon <= 77.8):
            # Skip coordinates outside Delhi NCR bounding box
            continue

        name = tags.get("name") or tags.get("name:en") or tags.get("ref") or f"OSM Feature #{el.get('id')}"
        category, is_critical = _classify_infra_feature(tags)

        parsed_features.append({
            "id": el.get("id"),
            "lat": lat,
            "lon": lon,
            "name": name,
            "category": category,
            "critical": is_critical,
            "source": "OSM"
        })

    _INFRA_CACHE = parsed_features


def get_infrastructure_geojson(
    bbox: Optional[List[float]] = None,
    max_features: int = 500
) -> Dict[str, Any]:
    """
    Returns viewport-filtered GeoJSON FeatureCollection of infrastructure assets.
    """
    _init_infrastructure_cache()
    t_start = time.time()

    if _INFRA_CACHE is None or len(_INFRA_CACHE) == 0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {"total": 0, "returned": 0, "source": "OSM"}
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
                "source": "OSM",
                "status": "outside_operating_area",
            },
        }

    min_lon, min_lat, max_lon, max_lat = clamped_bbox
    filtered = [
        item for item in _INFRA_CACHE
        if min_lon <= item["lon"] <= max_lon and min_lat <= item["lat"] <= max_lat
    ]

    total_matched = len(filtered)

    # Sort critical items first and cap output
    filtered_sorted = sorted(filtered, key=lambda x: (not x["critical"], x["name"]))[:max_features]

    features = []
    for item in filtered_sorted:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [item["lon"], item["lat"]]
            },
            "properties": {
                "id": item["id"],
                "name": item["name"],
                "category": item["category"],
                "critical": item["critical"],
                "source": item["source"]
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
            "source": "OSM"
        }
    }
