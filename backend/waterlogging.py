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
from shared_resources import get_shared_graph, get_shared_model_bundle, get_shared_spatial_trees, find_project_file
from pathlib import Path
from typing import Dict, Any, List, Tuple
from scipy.spatial import cKDTree
import rasterio

MODEL_PATH = find_project_file("models/aquag_model_v2.pkl")
METADATA_PATH = find_project_file("models/aquag_model_v2_metadata.json")
GRAPH_PATH = find_project_file("existing code/data/processed/delhi_road_graph_compact.npz")
DEM_PATH = find_project_file("existing code/data/raw/dem/delhi_aw3d30.tif")
DRAINS_PATH = find_project_file("existing code/data/raw/drainage/delhi_drains_mpd1976_full.geojson")
INFRA_PATH = find_project_file("existing code/data/raw/infrastructure/delhi_important_infrastructure.json")
POP_PATH = find_project_file("existing code/data/raw/population/delhi_districts_population_2011-3.geojson")

# ---------------------------------------------------------------------------
# Delhi Operating Area Boundary Helper
# ---------------------------------------------------------------------------
DELHI_OPERATING_BOUNDS = [76.80, 28.40, 77.40, 28.90]

def clamp_and_validate_bbox(bbox: List[float] | None) -> Tuple[List[float] | None, bool]:
    """
    Validates and clamps a requested bounding box to the Delhi study area.
    Returns (clamped_bbox, is_valid_inside_delhi).
    """
    if not bbox:
        return DELHI_OPERATING_BOUNDS, True
        
    if len(bbox) != 4:
        return None, False
        
    req_min_lon, req_min_lat, req_max_lon, req_max_lat = [float(x) for x in bbox]
    
    # Disjoint check
    if (
        req_max_lon <= DELHI_OPERATING_BOUNDS[0]
        or req_min_lon >= DELHI_OPERATING_BOUNDS[2]
        or req_max_lat <= DELHI_OPERATING_BOUNDS[1]
        or req_min_lat >= DELHI_OPERATING_BOUNDS[3]
    ):
        return None, False
        
    # Clamp to Delhi operating bounds
    c_min_lon = max(req_min_lon, DELHI_OPERATING_BOUNDS[0])
    c_min_lat = max(req_min_lat, DELHI_OPERATING_BOUNDS[1])
    c_max_lon = min(req_max_lon, DELHI_OPERATING_BOUNDS[2])
    c_max_lat = min(req_max_lat, DELHI_OPERATING_BOUNDS[3])
    
    if c_min_lon >= c_max_lon or c_min_lat >= c_max_lat:
        return None, False
        
    return [c_min_lon, c_min_lat, c_max_lon, c_max_lat], True

# ---------------------------------------------------------------------------
# Module Resources & Caches
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
_U_NODES = None

_DRAIN_KDTREE = None
_INFRA_KDTREE = None
_POP_KDTREE = None
_POP_TOTALS = None
_DEM_DATASET = None

_WATERLOGGING_CACHE: Dict[Tuple, Tuple[float, Dict[str, Any]]] = {}
_CACHE_MAX_SIZE = 20
_CACHE_TTL_SEC = 60.0


def _get_cached_waterlogging(cache_key: Tuple) -> Dict[str, Any] | None:
    now = time.time()
    if cache_key in _WATERLOGGING_CACHE:
        ts, result = _WATERLOGGING_CACHE[cache_key]
        if now - ts <= _CACHE_TTL_SEC:
            return result
        else:
            del _WATERLOGGING_CACHE[cache_key]
    return None


def _set_cached_waterlogging(cache_key: Tuple, result: Dict[str, Any]) -> None:
    now = time.time()
    if len(_WATERLOGGING_CACHE) >= _CACHE_MAX_SIZE:
        oldest_key = min(_WATERLOGGING_CACHE, key=lambda k: _WATERLOGGING_CACHE[k][0])
        del _WATERLOGGING_CACHE[oldest_key]
    _WATERLOGGING_CACHE[cache_key] = (now, result)


def _init_waterlogging_resources() -> None:
    """Initialize and cache all graph, model, spatial trees, and DEM resources once at startup."""
    global _MODEL, _METADATA, _FEATURE_ORDER, _CLASS_NAMES
    global _GRAPH_DATA, _LATS, _LONS, _OFFSETS, _TARGETS, _DISTANCES, _NODE_RISK_MULT, _U_NODES
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
        and _U_NODES is not None
    ):
        return

    # 1. Load Shared Model V2 and Metadata Singleton
    mb = get_shared_model_bundle()
    _MODEL = mb["model"]
    _METADATA = mb["metadata"]
    _FEATURE_ORDER = mb["feature_order"]
    _CLASS_NAMES = mb["class_names"]

    # 2. Load Shared Compact Road Graph Singleton
    g_data = get_shared_graph()
    _LATS = g_data["lat"]
    _LONS = g_data["lon"]
    _OFFSETS = g_data["offsets"]
    _TARGETS = g_data["targets"]
    _DISTANCES = g_data["distances"]
    _NODE_RISK_MULT = g_data["node_risk_mult"]
    _U_NODES = g_data["u_nodes"]
    _GRAPH_DATA = g_data

    # 3. Load Shared Spatial KDTrees
    st = get_shared_spatial_trees()
    _DRAIN_KDTREE = st["drain_tree"]
    _POP_KDTREE = st["pop_tree"]
    _POP_TOTALS = st["pop_totals"]
    _INFRA_KDTREE = st["infra_tree"]

    # 4. Load DEM Raster Dataset Handle
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

    # Validate & Clamp Bounding Box to Delhi Operating Bounds
    clamped_bbox, is_valid = clamp_and_validate_bbox(bbox)
    if not is_valid or clamped_bbox is None:
        return {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "total_candidate_segments": 0,
                "returned_segments": 0,
                "truncated": False,
                "scenario": scenario_clean,
                "timestep": str(timestep).upper().strip(),
                "bbox": bbox,
                "status": "outside_operating_area",
            },
        }
    min_lon, min_lat, max_lon, max_lat = clamped_bbox

    # 1. Check LRU Result Cache
    cache_key = (
        scenario_clean,
        str(timestep).upper().strip(),
        round(float(rainfall_1h), 1),
        round(float(rainfall_3h), 1),
        round(float(rainfall_6h), 1),
        round(float(recent_rainfall_intensity), 1),
        round(float(min_lon), 3),
        round(float(min_lat), 3),
        round(float(max_lon), 3),
        round(float(max_lat), 3),
    )
    cached_res = _get_cached_waterlogging(cache_key)
    if cached_res is not None:
        return cached_res

    # 2. Vectorized Node & Edge Bounding Box Filtering
    if _LONS is None or _LATS is None or _U_NODES is None:
        raise RuntimeError("Waterlogging graph coordinates failed to initialize")

    edge_mid_lats = (_LATS[_U_NODES] + _LATS[_TARGETS]) / 2.0
    edge_mid_lons = (_LONS[_U_NODES] + _LONS[_TARGETS]) / 2.0
    edge_mask = (edge_mid_lats >= min_lat) & (edge_mid_lats <= max_lat) & (edge_mid_lons >= min_lon) & (edge_mid_lons <= max_lon)
    valid_edge_indices = np.where(edge_mask)[0]

    total_candidates = len(valid_edge_indices)
    if total_candidates == 0:
        empty_res = {
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
        _set_cached_waterlogging(cache_key, empty_res)
        return empty_res

    # 3. Spatially Representative Grid Selection BEFORE KDTree Queries & DEM Sampling for Performance & RAM Safety
    truncated = False
    if total_candidates > max_segments:
        truncated = True
        grid_cols = 10
        grid_rows = 10
        cell_quota = max_segments // (grid_cols * grid_rows)

        v_lats = edge_mid_lats[valid_edge_indices]
        v_lons = edge_mid_lons[valid_edge_indices]
        v_risks = _NODE_RISK_MULT[_U_NODES[valid_edge_indices]]

        gx = np.clip(np.int32(((v_lons - min_lon) / (max_lon - min_lon)) * grid_cols), 0, grid_cols - 1)
        gy = np.clip(np.int32(((v_lats - min_lat) / (max_lat - min_lat)) * grid_rows), 0, grid_rows - 1)
        cell_ids = gy * grid_cols + gx

        sort_order = np.lexsort((-v_risks, cell_ids))
        sorted_edge_indices = valid_edge_indices[sort_order]
        sorted_cell_ids = cell_ids[sort_order]

        _, cell_start_indices, cell_counts = np.unique(sorted_cell_ids, return_index=True, return_counts=True)

        selected_indices = []
        for start, count in zip(cell_start_indices, cell_counts):
            take = min(count, cell_quota)
            selected_indices.extend(sorted_edge_indices[start : start + take])

        rem_capacity = max_segments - len(selected_indices)
        if rem_capacity > 0:
            sel_set = set(selected_indices)
            unselected = [idx for idx in sorted_edge_indices if idx not in sel_set]
            selected_indices.extend(unselected[:rem_capacity])

        top_indices = np.array(selected_indices[:max_segments], dtype=np.int32)
    else:
        top_indices = valid_edge_indices

    segment_u = _U_NODES[top_indices]
    segment_v = _TARGETS[top_indices]
    segment_dists = _DISTANCES[top_indices]

    mid_lats = (_LATS[segment_u] + _LATS[segment_v]) / 2.0
    mid_lons = (_LONS[segment_u] + _LONS[segment_v]) / 2.0
    midpoints_arr = np.column_stack([mid_lats, mid_lons]).astype(np.float32)

    N = len(segment_u)

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

    result = {
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

    _set_cached_waterlogging(cache_key, result)
    return result
