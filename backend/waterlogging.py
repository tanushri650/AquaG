"""
AquaG Street-Level Waterlogging Engine (Stage 12 Phase 1)

Computes viewport-filtered road segment flood severity and hydro-spatial waterlogging depth proxies
using Model V2 (XGBoost), DEM terrain elevation, MPD-1976 drainage proximity, and spatial exposure.

IMPORTANT SCIENTIFIC DISCLAIMER:
Calculated water depth values represent a DETERMINISTIC HYDRO-SPATIAL PROXY derived from
Model V2 flood severity probabilities and terrain depression features.
They DO NOT represent physically measured sensor water depth readings.
"""

from __future__ import annotations
import math
import time
import json
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from scipy.spatial import cKDTree
import rasterio

# ---------------------------------------------------------------------------
# Path Resolutions
# ---------------------------------------------------------------------------
def find_project_file(relative_path: str) -> Path:
    this_dir = Path(__file__).resolve().parent
    candidate_roots = [
        this_dir.parent.parent,
        this_dir.parent,
        this_dir,
        Path.cwd(),
        Path.cwd().parent,
    ]
    clean_rel = relative_path.replace("\\", "/").strip("/")
    rel_parts = clean_rel.split("/")

    for root in candidate_roots:
        target = root.joinpath(*rel_parts)
        if target.exists():
            return target
        if rel_parts[0] == "existing code":
            target_stripped = root.joinpath(*rel_parts[1:])
            if target_stripped.exists():
                return target_stripped
        target_added = root.joinpath("existing code", *rel_parts)
        if target_added.exists():
            return target_added

    return candidate_roots[0].joinpath(*rel_parts)

MODEL_PATH = find_project_file("models/aquag_model_v2.pkl")
METADATA_PATH = find_project_file("models/aquag_model_v2_metadata.json")
GRAPH_PATH = find_project_file("existing code/data/processed/delhi_road_graph_compact.npz")
DEM_PATH = find_project_file("existing code/data/raw/dem/delhi_aw3d30.tif")
DRAINS_PATH = find_project_file("existing code/data/raw/drainage/delhi_drains_mpd1976_full.geojson")
INFRA_PATH = find_project_file("existing code/data/raw/infrastructure/delhi_important_infrastructure.json")
POP_PATH = find_project_file("existing code/data/raw/population/delhi_districts_population_2011-3.geojson")

# ---------------------------------------------------------------------------
# Lazy Loaded Singletons
# ---------------------------------------------------------------------------
_MODEL = None
_METADATA = None
_FEATURE_ORDER = None
_CLASS_NAMES = None

_GRAPH_DATA = None
_LATS = None
_LONS = None
_OFFSETS = None
_TARGETS = None
_DISTANCES = None
_NODE_RISK_MULT = None

_DRAIN_KDTREE = None
_INFRA_KDTREE = None
_POP_KDTREE = None
_POP_TOTALS = None
_DEM_DATASET = None


def _init_waterlogging_resources() -> None:
    """Initialize and cache all graph, model, spatial trees, and DEM resources once at startup."""
    global _MODEL, _METADATA, _FEATURE_ORDER, _CLASS_NAMES
    global _GRAPH_DATA, _LATS, _LONS, _OFFSETS, _TARGETS, _DISTANCES, _NODE_RISK_MULT
    global _DRAIN_KDTREE, _INFRA_KDTREE, _POP_KDTREE, _POP_TOTALS, _DEM_DATASET

    if (
        _MODEL is not None 
        and _GRAPH_DATA is not None 
        and _LATS is not None 
        and _LONS is not None
        and _OFFSETS is not None
        and _TARGETS is not None
        and _DISTANCES is not None
        and _NODE_RISK_MULT is not None
    ):
        return

    # 1. Load Model V2 and Metadata
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(f"Model V2 or metadata artifact missing for waterlogging engine: {MODEL_PATH}")
    
    _MODEL = joblib.load(MODEL_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        _METADATA = json.load(f)
    
    _FEATURE_ORDER = _METADATA["feature_order"]
    _CLASS_NAMES = _METADATA.get("class_names", ["High", "Low", "Medium"])

    # 2. Load Compact Road Graph
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Compact road graph missing: {GRAPH_PATH}")
    
    _GRAPH_DATA = np.load(GRAPH_PATH)
    _LATS = np.ascontiguousarray(_GRAPH_DATA["lat"], dtype=np.float32)
    _LONS = np.ascontiguousarray(_GRAPH_DATA["lon"], dtype=np.float32)
    _OFFSETS = np.ascontiguousarray(_GRAPH_DATA["offsets"], dtype=np.int32)
    _TARGETS = np.ascontiguousarray(_GRAPH_DATA["targets"], dtype=np.int32)
    _DISTANCES = np.ascontiguousarray(_GRAPH_DATA["distances"], dtype=np.float32)
    _NODE_RISK_MULT = np.ascontiguousarray(_GRAPH_DATA["node_risk_mult"], dtype=np.float32)

    # 3. Build Drains Spatial KDTree
    if DRAINS_PATH.exists():
        try:
            drn_gdf = gpd.read_file(DRAINS_PATH)
            drn_coords = np.column_stack([drn_gdf.geometry.y, drn_gdf.geometry.x]).astype(np.float32)
            _DRAIN_KDTREE = cKDTree(drn_coords)
        except Exception:
            _DRAIN_KDTREE = None

    # 4. Build Population Spatial KDTree
    if POP_PATH.exists():
        try:
            pop_gdf = gpd.read_file(POP_PATH)
            pop_coords = np.column_stack([pop_gdf.geometry.y, pop_gdf.geometry.x]).astype(np.float32)
            _POP_TOTALS = pop_gdf["population_total"].values
            _POP_KDTREE = cKDTree(pop_coords)
        except Exception:
            _POP_KDTREE = None
            _POP_TOTALS = None

    # 5. Build Infrastructure Spatial KDTree
    if INFRA_PATH.exists():
        try:
            with open(INFRA_PATH, "r", encoding="utf-8") as f:
                infra_raw = json.load(f)
            
            infra_coords = []
            for el in infra_raw.get("elements", []):
                lat = el.get("lat") or el.get("center", {}).get("lat")
                lon = el.get("lon") or el.get("center", {}).get("lon")
                if lat is not None and lon is not None:
                    infra_coords.append([lat, lon])
            
            if infra_coords:
                _INFRA_KDTREE = cKDTree(np.array(infra_coords, dtype=np.float32))
        except Exception:
            _INFRA_KDTREE = None

    # 6. Load DEM Raster Dataset Handle
    if DEM_PATH.exists():
        try:
            _DEM_DATASET = rasterio.open(DEM_PATH)
        except Exception:
            _DEM_DATASET = None


def calculate_water_depth_proxy(
    high_prob: float,
    medium_prob: float,
    low_prob: float,
    elevation_m: float,
    distance_to_drain_m: float,
    intensity: float,
) -> Tuple[float, str, str]:
    """
    Deterministic hydro-spatial waterlogging depth proxy (cm).
    
    Combines Model V2 severity probabilities, elevation depression bonus,
    drainage proximity, and rainfall intensity into a bounded, non-negative depth proxy.
    """
    prob_base = (high_prob * 65.0) + (medium_prob * 25.0) + (low_prob * 2.0)
    depression_bonus = max(0.0, 218.0 - elevation_m) * 1.8
    drain_bonus = max(0.0, (150.0 - distance_to_drain_m) / 150.0) * 12.0
    intensity_factor = min(1.5, max(0.7, intensity / 20.0))
    
    depth_cm = round(max(0.0, (prob_base + depression_bonus + drain_bonus) * intensity_factor), 1)
    
    if depth_cm > 100.0:
        severity = "High"
        depth_band = ">100 cm"
    elif depth_cm > 50.0:
        severity = "High"
        depth_band = "50-100 cm"
    elif depth_cm > 25.0:
        severity = "Medium"
        depth_band = "25-50 cm"
    elif depth_cm > 10.0:
        severity = "Medium"
        depth_band = "10-25 cm"
    else:
        severity = "Low"
        depth_band = "0-10 cm"
        
    return depth_cm, severity, depth_band


def _get_timestep_multiplier(timestep: str) -> float:
    """Return deterministic rainfall progression factor for T+0..T+3 timesteps."""
    step_clean = timestep.upper().strip()
    if step_clean in ("T+0", "NOW", "0"):
        return 0.60
    elif step_clean in ("T+1", "+1H", "1"):
        return 1.00
    elif step_clean in ("T+2", "+2H", "2"):
        return 1.35
    elif step_clean in ("T+3", "+3H", "3"):
        return 1.65
    return 1.00


def get_street_waterlogging_geojson(
    scenario: str = "MODERATE",
    timestep: str = "T+0",
    rainfall_1h: float | None = None,
    rainfall_3h: float | None = None,
    rainfall_6h: float | None = None,
    recent_rainfall_intensity: float | None = None,
    bbox: List[float] | None = None,
    max_segments: int = 2500,
) -> Dict[str, Any]:
    """
    Generate viewport-filtered GeoJSON FeatureCollection of waterlogged road segments.
    
    Memory and execution optimized for Render 512 MB RAM deployment.
    """
    _init_waterlogging_resources()
    t_start = time.time()

    scenario_clean = str(scenario).upper().strip() if scenario else "MODERATE"
    preset_rain = {
        "NORMAL": (10.0, 20.0, 30.0, 5.0),
        "LOW": (10.0, 20.0, 30.0, 5.0),
        "MODERATE": (45.0, 85.0, 130.0, 22.5),
        "MEDIUM": (45.0, 85.0, 130.0, 22.5),
        "HEAVY": (75.0, 130.0, 180.0, 37.5),
        "HIGH": (75.0, 130.0, 180.0, 37.5),
        "EXTREME": (110.0, 180.0, 250.0, 55.0),
    }.get(scenario_clean, (45.0, 85.0, 130.0, 22.5))

    if rainfall_1h is None:
        rainfall_1h = preset_rain[0]
    if rainfall_3h is None:
        rainfall_3h = preset_rain[1]
    if rainfall_6h is None:
        rainfall_6h = preset_rain[2]
    if recent_rainfall_intensity is None:
        recent_rainfall_intensity = preset_rain[3]

    # Validate Bounding Box
    if not bbox or len(bbox) != 4:
        raise ValueError("Invalid bounding box: must be [min_lon, min_lat, max_lon, max_lat]")
    
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("Invalid bounding box bounds: min values must be strictly less than max values")

    # 1. Filter Nodes within Viewport Bounding Box
    if _LONS is None or _LATS is None:
        raise RuntimeError("Waterlogging graph coordinates failed to initialize")

    node_mask = (_LONS >= min_lon) & (_LONS <= max_lon) & (_LATS >= min_lat) & (_LATS <= max_lat)
    valid_nodes = np.where(node_mask)[0]

    if len(valid_nodes) == 0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "total_candidate_segments": 0,
                "returned_segments": 0,
                "truncated": False,
                "scenario": scenario,
                "timestep": timestep,
                "bbox": bbox,
            },
        }

    # 2. Extract Edges for Filtered Nodes
    segment_u = []
    segment_v = []
    segment_midpoints = []
    segment_dists = []

    for u in valid_nodes:
        start_edge = _OFFSETS[u]
        end_edge = _OFFSETS[u + 1]
        for e_idx in range(start_edge, end_edge):
            v = _TARGETS[e_idx]
            u_lat, u_lon = _LATS[u], _LONS[u]
            v_lat, v_lon = _LATS[v], _LONS[v]
            
            mid_lat = (u_lat + v_lat) / 2.0
            mid_lon = (u_lon + v_lon) / 2.0
            
            segment_u.append(u)
            segment_v.append(v)
            segment_midpoints.append([mid_lat, mid_lon])
            segment_dists.append(float(_DISTANCES[e_idx]))

    total_candidates = len(segment_u)
    
    # 3. Truncate / Cap Segments for Performance & RAM Safety
    truncated = False
    if total_candidates > max_segments:
        truncated = True
        # Sort/prioritize segments with higher node risk multiplier or lower node index
        risks = _NODE_RISK_MULT[np.array(segment_u)]
        top_indices = np.argsort(-risks)[:max_segments]
        
        segment_u = [segment_u[i] for i in top_indices]
        segment_v = [segment_v[i] for i in top_indices]
        segment_midpoints = [segment_midpoints[i] for i in top_indices]
        segment_dists = [segment_dists[i] for i in top_indices]

    N = len(segment_u)
    midpoints_arr = np.array(segment_midpoints, dtype=np.float32)

    # 4. Batch Spatial KDTree Queries
    # Drain Distance Query
    if _DRAIN_KDTREE is not None:
        drain_dists_deg, _ = _DRAIN_KDTREE.query(midpoints_arr)
        drain_dists_m = (drain_dists_deg * 111000.0).astype(np.float32)
    else:
        drain_dists_m = np.full(N, 500.0, dtype=np.float32)

    # Infrastructure Distance Query
    if _INFRA_KDTREE is not None:
        infra_dists_deg, _ = _INFRA_KDTREE.query(midpoints_arr)
        infra_dists_m = (infra_dists_deg * 111000.0).astype(np.float32)
    else:
        infra_dists_m = np.full(N, 300.0, dtype=np.float32)

    crit_infra_flags = (infra_dists_m < 250.0).astype(int)

    # Population Exposure Query
    if _POP_KDTREE is not None and _POP_TOTALS is not None:
        _, pop_indices = _POP_KDTREE.query(midpoints_arr)
        pop_totals = _POP_TOTALS[pop_indices].astype(np.float32)
    else:
        pop_totals = np.full(N, 100000.0, dtype=np.float32)

    # DEM Elevation Query via Raster Batch Sampling
    if _DEM_DATASET is not None:
        try:
            sample_coords = [(float(pt[1]), float(pt[0])) for pt in midpoints_arr]
            elevations = [float(vals[0]) for vals in _DEM_DATASET.sample(sample_coords)]
            elevations_m = np.array([e if e > 0 else 215.0 for e in elevations], dtype=np.float32)
        except Exception:
            elevations_m = np.full(N, 215.0, dtype=np.float32)
    else:
        elevations_m = np.full(N, 215.0, dtype=np.float32)

    # Slope Approximation (degree)
    slopes = np.ones(N, dtype=np.float32)

    # 5. Timestep Rainfall Scaling
    ts_mult = _get_timestep_multiplier(timestep)
    eff_r1h = rainfall_1h * ts_mult
    eff_r3h = rainfall_3h * ts_mult
    eff_r6h = rainfall_6h * ts_mult
    eff_intensity = recent_rainfall_intensity * ts_mult

    # 6. Construct Batched Feature Matrix X (N x 11)
    # Feature Order: ['rainfall_1h', 'rainfall_3h', 'rainfall_6h', 'recent_rainfall_intensity',
    #                 'elevation', 'slope', 'distance_to_drain', 'distance_to_road',
    #                 'distance_to_infra', 'population_total', 'critical_infra_flag']
    X = np.column_stack([
        np.full(N, eff_r1h, dtype=np.float32),
        np.full(N, eff_r3h, dtype=np.float32),
        np.full(N, eff_r6h, dtype=np.float32),
        np.full(N, eff_intensity, dtype=np.float32),
        elevations_m,
        slopes,
        drain_dists_m,
        np.full(N, 15.0, dtype=np.float32),  # distance_to_road
        infra_dists_m,
        pop_totals,
        crit_infra_flags,
    ])

    feature_df = pd.DataFrame(X, columns=_FEATURE_ORDER)

    # 7. Single Batched Model V2 Inference
    probs_matrix = _MODEL.predict_proba(feature_df)

    # Class order mapping from metadata: Class 0 = High, Class 1 = Low, Class 2 = Medium
    high_probs = probs_matrix[:, 0]
    low_probs = probs_matrix[:, 1]
    med_probs = probs_matrix[:, 2]

    # 8. Build GeoJSON Features
    features = []
    for i in range(N):
        u = segment_u[i]
        v = segment_v[i]
        
        hp = float(high_probs[i])
        mp = float(med_probs[i])
        lp = float(low_probs[i])
        
        elev = float(elevations_m[i])
        d_drain = float(drain_dists_m[i])
        d_infra = float(infra_dists_m[i])
        pop = float(pop_totals[i])
        c_flag = int(crit_infra_flags[i])
        dist_m = float(segment_dists[i])
        
        depth_cm, severity, depth_band = calculate_water_depth_proxy(
            high_prob=hp,
            medium_prob=mp,
            low_prob=lp,
            elevation_m=elev,
            distance_to_drain_m=d_drain,
            intensity=eff_intensity,
        )

        coord_start = [float(_LONS[u]), float(_LATS[u])]
        coord_end = [float(_LONS[v]), float(_LATS[v])]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [coord_start, coord_end],
            },
            "properties": {
                "road_id": f"road_{u}_{v}",
                "start_node": int(u),
                "end_node": int(v),
                "distance_m": round(dist_m, 1),
                "water_depth_cm": depth_cm,
                "severity": severity,
                "depth_band": depth_band,
                "low_probability": round(lp, 4),
                "medium_probability": round(mp, 4),
                "high_probability": round(hp, 4),
                "elevation_m": round(elev, 1),
                "distance_to_drain_m": round(d_drain, 1),
                "distance_to_infra_m": round(d_infra, 1),
                "population_exposure": int(pop),
                "critical_infra_flag": c_flag,
                "rainfall_1h": round(eff_r1h, 1),
                "rainfall_3h": round(eff_r3h, 1),
                "rainfall_6h": round(eff_r6h, 1),
                "recent_rainfall_intensity": round(eff_intensity, 1),
                "timestep": timestep,
                "basis": "hydro_spatial_proxy",
            },
        }
        features.append(feature)

    exec_ms = round((time.time() - t_start) * 1000.0, 1)

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_candidate_segments": total_candidates,
            "returned_segments": len(features),
            "truncated": truncated,
            "scenario": scenario,
            "timestep": timestep,
            "bbox": bbox,
            "execution_ms": exec_ms,
            "basis": "hydro_spatial_proxy",
        },
    }
