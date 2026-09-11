"""
AquaG Population Priority Engine (Stage 12 Phase 3B)

Combines viewport street-level waterlogging results with district-level 2011 Census
population exposure and critical infrastructure spatial flags into a deterministic
response priority score (0-100) and classification (LOW, MEDIUM, HIGH, CRITICAL).

IMPORTANT SCIENTIFIC DISCLAIMER:
Population exposure represents a DISTRICT-LEVEL POPULATION EXPOSURE PROXY derived from
2011 Census district centroids. It DOES NOT represent exact counts of individuals on a road.
"""

from __future__ import annotations
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from scipy.spatial import cKDTree

from shared_resources import get_shared_spatial_trees, find_project_file
from waterlogging import get_street_waterlogging_geojson

# ---------------------------------------------------------------------------
# Path Resolutions & Resources
# ---------------------------------------------------------------------------
POP_PATH = find_project_file("existing code/data/raw/population/delhi_districts_population_2011-3.geojson")

_POP_KDTREE: Optional[cKDTree] = None
_POP_NAMES: Optional[np.ndarray] = None
_POP_TOTALS: Optional[np.ndarray] = None


def _init_population_resources() -> None:
    """Lazy load district population centroids and build spatial KDTree."""
    global _POP_KDTREE, _POP_NAMES, _POP_TOTALS
    if _POP_KDTREE is not None:
        return

    st = get_shared_spatial_trees()
    _POP_KDTREE = st["pop_tree"]
    _POP_TOTALS = st["pop_totals"]
    _POP_NAMES = st["pop_names"]


def calculate_population_priority_score(
    water_depth_cm: float,
    severity: str,
    population_exposure: float,
    critical_infra_flag: int,
) -> tuple[float, str]:
    """
    Deterministic response priority score (0.0 - 100.0) and level classification.
    
    Components:
    1. Depth hazard component (max 40 pts): min(40.0, (water_depth_cm / 100.0) * 40.0)
    2. Population exposure component (max 25 pts): min(25.0, (population_exposure / 3656539.0) * 25.0)
    3. Critical infrastructure flag component (max 20 pts): 20.0 if critical_infra_flag == 1 else 0.0
    4. Model V2 severity component (max 15 pts): High=15.0, Medium=8.0, Low=2.0
    """
    depth_pts = min(40.0, (max(0.0, water_depth_cm) / 100.0) * 40.0)
    pop_pts = min(25.0, (max(0.0, population_exposure) / 3656539.0) * 25.0)
    infra_pts = 20.0 if critical_infra_flag == 1 else 0.0
    
    sev_clean = str(severity).capitalize()
    if sev_clean == "High":
        sev_pts = 15.0
    elif sev_clean == "Medium":
        sev_pts = 8.0
    else:
        sev_pts = 2.0

    score = round(min(100.0, depth_pts + pop_pts + infra_pts + sev_pts), 1)

    if score >= 75.0:
        level = "CRITICAL"
    elif score >= 50.0:
        level = "HIGH"
    elif score >= 25.0:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level


def get_population_priority_geojson(
    scenario: str = "NORMAL",
    timestep: str = "T+0",
    rainfall_1h: Optional[float] = None,
    rainfall_3h: Optional[float] = None,
    rainfall_6h: Optional[float] = None,
    recent_rainfall_intensity: Optional[float] = None,
    bbox: Optional[List[float]] = None,
    max_features: int = 2500,
) -> Dict[str, Any]:
    """
    Returns GeoJSON FeatureCollection of road segments with population response priority scores.
    """
    _init_population_resources()
    t_start = time.time()

    timestep = str(timestep).upper().replace(" ", "+").strip()
    scenario = str(scenario).upper().strip()

    # 1. Fetch Waterlogging GeoJSON for specified scenario, timestep, and bbox

    wl_geojson = get_street_waterlogging_geojson(
        scenario=scenario,
        timestep=timestep,
        rainfall_1h=rainfall_1h,
        rainfall_3h=rainfall_3h,
        rainfall_6h=rainfall_6h,
        recent_rainfall_intensity=recent_rainfall_intensity,
        bbox=bbox,
        max_segments=max_features,
    )

    wl_features = wl_geojson.get("features", [])
    if not wl_features:
        return {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "total_features": 0,
                "scenario": scenario,
                "timestep": timestep,
                "basis": "district_population_exposure_proxy",
            },
        }

    # Extract midpoints for spatial district mapping if KDTree available
    midpoints = []
    for f in wl_features:
        coords = f["geometry"]["coordinates"]
        mid_lon = (coords[0][0] + coords[1][0]) / 2.0
        mid_lat = (coords[0][1] + coords[1][1]) / 2.0
        midpoints.append([mid_lat, mid_lon])

    midpoints_arr = np.array(midpoints, dtype=np.float32)

    if _POP_KDTREE is not None and _POP_NAMES is not None and _POP_TOTALS is not None:
        _, pop_indices = _POP_KDTREE.query(midpoints_arr)
        matched_districts = _POP_NAMES[pop_indices]
        matched_pops = _POP_TOTALS[pop_indices]
    else:
        matched_districts = np.full(len(wl_features), "Delhi District")
        matched_pops = np.full(len(wl_features), 1500000.0, dtype=np.float32)

    priority_features = []
    for i, f in enumerate(wl_features):
        props = f["properties"]
        water_depth_cm = float(props.get("water_depth_cm", 0.0))
        severity = str(props.get("severity", "Low"))
        crit_flag = int(props.get("critical_infra_flag", 0))
        
        district_name = str(matched_districts[i])
        pop_exp = float(matched_pops[i])

        score, priority_level = calculate_population_priority_score(
            water_depth_cm=water_depth_cm,
            severity=severity,
            population_exposure=pop_exp,
            critical_infra_flag=crit_flag,
        )

        p_feature = {
            "type": "Feature",
            "geometry": f["geometry"],
            "properties": {
                "road_id": props.get("road_id", f"road_{i}"),
                "water_depth_cm": water_depth_cm,
                "severity": severity,
                "population_exposure": int(pop_exp),
                "critical_infra_flag": crit_flag,
                "priority_score": score,
                "priority_level": priority_level,
                "district": district_name,
                "elevation_m": props.get("elevation_m"),
                "distance_to_drain_m": props.get("distance_to_drain_m"),
                "timestep": timestep,
                "scenario": scenario,
                "basis": "district_population_exposure_proxy",
            },
        }
        priority_features.append(p_feature)

    exec_ms = round((time.time() - t_start) * 1000.0, 1)

    return {
        "type": "FeatureCollection",
        "features": priority_features,
        "metadata": {
            "total_features": len(priority_features),
            "scenario": scenario,
            "timestep": timestep,
            "bbox": bbox,
            "execution_ms": exec_ms,
            "basis": "district_population_exposure_proxy",
        },
    }
