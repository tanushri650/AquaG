import json
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree

from waterlogging import find_project_file, clamp_and_validate_bbox, DELHI_OPERATING_BOUNDS

# ---------------------------------------------------------------------------
# Path Resolutions & Constants
# ---------------------------------------------------------------------------
INFRA_PATH = find_project_file("existing code/data/raw/infrastructure/delhi_important_infrastructure.json")

_INFRA_CACHE: Optional[List[Dict[str, Any]]] = None


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return approximate distance between two coordinates in metres."""
    earth_radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * earth_radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _classify_infra_feature(tags: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Classify OSM tags into approved AquaG critical categories using the refined audit whitelist:
    1. POWER: Grid substations and power plants ONLY. Excludes generators, transformers, poles, lines, towers, switches.
    2. MEDICAL: Major hospitals & trauma centers ONLY. Excludes dispensaries, clinics, doctors offices, dental clinics.
    3. FIRE: Genuine fire stations ONLY. Excludes bus stops named after fire stations.
    4. POLICE: Main 24/7 police stations & headquarters ONLY. Excludes police posts, chowkis, pickets, booths, facilitation booths.
    5. TRANSPORT: Main Metro stations, railway stations, airports, ISBT bus terminals ONLY. Excludes exit gates, gate numbers, bus stops, platform nodes.
    6. WATER/UTILITY: Water/Sewage Treatment Plants & major pumping stations ONLY. Excludes overhead water towers/tanks. Avoids pump layer duplication.
    """
    amenity = str(tags.get("amenity", "")).lower()
    power = str(tags.get("power", "")).lower()
    railway = str(tags.get("railway", "")).lower()
    man_made = str(tags.get("man_made", "")).lower()
    healthcare = str(tags.get("healthcare", "")).lower()
    emergency = str(tags.get("emergency", "")).lower()
    building = str(tags.get("building", "")).lower()
    aeroway = str(tags.get("aeroway", "")).lower()
    public_transport = str(tags.get("public_transport", "")).lower()
    highway = str(tags.get("highway", "")).lower()
    station = str(tags.get("station", "")).lower()
    name = str(tags.get("name", "")).lower()

    # GLOBAL EXCLUSION GUARD: Discard all bus stop platform nodes, transit stop positions, subway entrance gates
    if highway == "bus_stop" or public_transport in ("platform", "stop_position", "stop") or railway in ("subway_entrance", "halt") or "bus stop" in name:
        return "NON_CRITICAL", False

    # 1. POWER: Grid substations and power plants ONLY (Strict exclusions: generator, transformer, pole, tower, line)
    if power in ("generator", "transformer", "pole", "tower", "line", "minor_line", "cable", "cables", "switch", "terminal") or any(x in name for x in ("generator", "transformer", "pole", "tower", "distribution transformer")):
        pass
    elif power in ("substation", "plant", "station") or ("substation" in name and "bus" not in name) or "power station" in name or "power plant" in name:
        return "POWER", True

    # 2. MEDICAL: Major Hospitals ONLY (Strict exclusions: clinic, dispensary, doctors office, dental, veterinary, pharmacy)
    if amenity in ("clinic", "doctors", "veterinary", "pharmacy") or healthcare in ("clinic", "doctor", "pharmacy", "centre") or any(x in name for x in ("dispensary", "clinic", "dental", "doctor office", "doctors office", "health centre", "health center", "veterinary", "sanitiser", "diagnostic", "lab")):
        pass
    elif amenity == "hospital" or healthcare == "hospital" or "hospital" in name:
        return "MEDICAL", True

    # 3. FIRE: Genuine Fire Stations ONLY
    if amenity == "fire_station" or emergency == "fire_station" or "fire station" in name or "fire brigade" in name:
        return "FIRE", True

    # 4. POLICE: Main Police Stations ONLY (Strict exclusions: post, chowki, chauki, picket, booth, facilitation, outpost)
    if any(x in name for x in ("post", "chowki", "chauki", "picket", "booth", "facilitation", "outpost")):
        pass
    elif amenity == "police" or building == "police" or "police station" in name or "thana" in name:
        return "POLICE", True

    # 5. TRANSPORT: Metro/Railway Stations, Terminals, Airports ONLY (Strict exclusions: exit gates, gate numbers)
    if any(x in name for x in ("gate no", "gate number", "gate 1", "gate 2", "gate 3", "gate 4", "gate 5", "gate 6")):
        pass
    elif station in ("subway", "station") or railway in ("station", "subway") or aeroway in ("terminal", "aerodrome") or amenity == "bus_station" or "metro station" in name or "railway station" in name or "isbt" in name or "airport" in name:
        return "TRANSPORT", True

    # 6. WATER / UTILITY: Water/Sewage Treatment Plants & Pumping Stations ONLY (Strict exclusions: water towers, overhead tanks)
    if man_made == "water_tower" or any(x in name for x in ("water tower", "overhead tank", "water tank")):
        pass
    elif man_made in ("water_works", "wastewater_plant", "pumping_station") or "water treatment" in name or "sewage treatment" in name or "pumping station" in name or "wtp" in name or "stp" in name:
        return "WATER/UTILITY", True

    return "NON_CRITICAL", False


def _init_infrastructure_cache() -> None:
    """Lazy load OSM infrastructure dataset once into memory, retaining full source records."""
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

        category, is_critical = _classify_infra_feature(tags)
        name = tags.get("name") or tags.get("name:en") or tags.get("ref") or f"Facility #{el.get('id')}"

        parsed_features.append({
            "id": el.get("id"),
            "lat": lat,
            "lon": lon,
            "name": name,
            "category": category,
            "critical_asset": is_critical,
            "source": "OSM"
        })

    _INFRA_CACHE = parsed_features


def get_infrastructure_geojson(
    scenario: str = "MODERATE",
    timestep: str = "T+0",
    rainfall_1h: Optional[float] = None,
    rainfall_3h: Optional[float] = None,
    rainfall_6h: Optional[float] = None,
    recent_rainfall_intensity: Optional[float] = None,
    bbox: Optional[List[float]] = None,
    max_features: int = 500
) -> Dict[str, Any]:
    """
    Returns viewport-filtered GeoJSON FeatureCollection of FLOOD-THREATENED critical infrastructure assets.
    Renders ONLY features where critical_asset == True AND forecast_threat == True.
    """
    _init_infrastructure_cache()
    t_start = time.time()

    if _INFRA_CACHE is None or len(_INFRA_CACHE) == 0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {"total_matched": 0, "returned_features": 0, "source": "OSM"}
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
                "source": "OSM",
                "status": "outside_operating_area",
            },
        }

    min_lon, min_lat, max_lon, max_lat = clamped_bbox
    viewport_infra = [
        item for item in _INFRA_CACHE
        if min_lon <= item["lon"] <= max_lon and min_lat <= item["lat"] <= max_lat
    ]
    total_infra_in_viewport = len(viewport_infra)

    # Filter 1: Retain ONLY genuinely critical assets in viewport
    viewport_critical = [item for item in viewport_infra if item["critical_asset"] and item["category"] != "NON_CRITICAL"]

    if len(viewport_critical) == 0:
        exec_ms = round((time.time() - t_start) * 1000.0, 1)
        return {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "total_infra_in_viewport": total_infra_in_viewport,
                "total_critical_in_viewport": 0,
                "total_matched": 0,
                "returned_features": 0,
                "execution_ms": exec_ms,
                "source": "OSM"
            }
        }

    # Fetch current forecast street waterlogging features for spatial flood filtering
    from waterlogging import get_street_waterlogging_geojson
    wl_geojson = get_street_waterlogging_geojson(
        scenario=scenario,
        timestep=timestep,
        rainfall_1h=rainfall_1h,
        rainfall_3h=rainfall_3h,
        rainfall_6h=rainfall_6h,
        recent_rainfall_intensity=recent_rainfall_intensity,
        bbox=clamped_bbox,
        max_segments=3000,
    )

    wl_features = wl_geojson.get("features", [])

    # Extract flooded road points and depths (>10cm depth)
    flooded_points = []
    flooded_depths = []

    for wf in wl_features:
        props = wf.get("properties", {})
        depth = float(props.get("water_depth_cm", 0.0))
        if depth > 10.0:  # Medium, High, or Critical waterlogging depth
            geom = wf.get("geometry", {})
            coords = geom.get("coordinates", [])
            for pt in coords:
                if len(pt) >= 2:
                    flooded_points.append([pt[1], pt[0]])  # [lat, lon]
                    flooded_depths.append(depth)

    threatened_features = []

    if len(flooded_points) > 0:
        flooded_coords_np = np.array(flooded_points, dtype=np.float32)
        flooded_kdtree = cKDTree(flooded_coords_np)

        for item in viewport_critical:
            infra_pt = [item["lat"], item["lon"]]
            dist_deg, idx = flooded_kdtree.query(infra_pt)
            nearest_lat, nearest_lon = flooded_coords_np[idx]
            dist_m = _haversine_meters(item["lat"], item["lon"], float(nearest_lat), float(nearest_lon))
            nearest_depth = float(flooded_depths[idx])

            # Conservative operational threshold check:
            # - Depth >100cm within 500m
            # - Depth >25cm within 400m
            # - Depth >10cm within 250m
            is_threatened = (
                (nearest_depth > 100.0 and dist_m <= 500.0)
                or (nearest_depth > 25.0 and dist_m <= 400.0)
                or (nearest_depth > 10.0 and dist_m <= 250.0)
            )

            if is_threatened:
                exposure_basis = (
                    "Critical flood proximity" if nearest_depth > 100.0
                    else "High flood proximity" if nearest_depth > 25.0
                    else "Waterlogging proximity"
                )

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
                        "critical": True,
                        "critical_asset": True,
                        "forecast_threat": True,
                        "forecast_status": "Forecast Threatened",
                        "flood_exposure_basis": exposure_basis,
                        "nearest_water_depth_cm": round(nearest_depth, 1),
                        "proximity_distance_m": round(dist_m, 1),
                        "timestep": timestep,
                        "scenario": scenario,
                        "source": item["source"]
                    }
                }
                threatened_features.append(feature)

    # Sort threatened items by proximity distance
    threatened_sorted = sorted(
        threatened_features,
        key=lambda x: x["properties"]["proximity_distance_m"]
    )[:max_features]

    exec_ms = round((time.time() - t_start) * 1000.0, 1)

    return {
        "type": "FeatureCollection",
        "features": threatened_sorted,
        "metadata": {
            "total_infra_in_viewport": total_infra_in_viewport,
            "total_critical_in_viewport": len(viewport_critical),
            "total_matched": len(threatened_features),
            "returned_features": len(threatened_sorted),
            "max_features_cap": max_features,
            "timestep": timestep,
            "scenario": scenario,
            "bbox": bbox,
            "execution_ms": exec_ms,
            "source": "OSM"
        }
    }
