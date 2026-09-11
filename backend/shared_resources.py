"""
AquaG Shared Singleton Resource Manager
Optimized memory architecture for Render 512 MB deployment limit.

Key Features:
- Single-instance loading of compact road graph NumPy arrays (shared between router & waterlogging)
- Memory-efficient FastNodeIndex using np.searchsorted (saves ~25 MB over Python dict)
- Single-instance loading of Model V2 XGBoost bundle
- Zero-GeoPandas JSON parsing for spatial KDTrees (saves ~160 MB overhead)
"""

from __future__ import annotations
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Path Resolution Helper
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

# Paths
MODEL_PATH = find_project_file("models/aquag_model_v2.pkl")
METADATA_PATH = find_project_file("models/aquag_model_v2_metadata.json")
GRAPH_PATH = find_project_file("existing code/data/processed/delhi_road_graph_compact.npz")
DRAINS_PATH = find_project_file("existing code/data/raw/drainage/delhi_drains_mpd1976_full.geojson")
INFRA_PATH = find_project_file("existing code/data/raw/infrastructure/delhi_important_infrastructure.json")
POP_PATH = find_project_file("existing code/data/raw/population/delhi_districts_population_2011-3.geojson")

# ---------------------------------------------------------------------------
# Fast Binary-Search Node Index (Replaces 20 MB Python dict with 5.2 MB arrays)
# ---------------------------------------------------------------------------
class FastNodeIndex:
    """
    Memory-efficient replacement for dict mapping OSM Node ID to graph array index.
    Uses np.searchsorted over argsorted array.
    """
    def __init__(self, node_ids: np.ndarray):
        self.sort_order = np.argsort(node_ids).astype(np.int32)
        self.sorted_ids = node_ids[self.sort_order]
        self.n = len(self.sorted_ids)

    def get(self, osm_id: int, default: Optional[int] = None) -> Optional[int]:
        pos = int(np.searchsorted(self.sorted_ids, osm_id))
        if pos < self.n and self.sorted_ids[pos] == osm_id:
            return int(self.sort_order[pos])
        return default

    def __getitem__(self, osm_id: int) -> int:
        val = self.get(osm_id)
        if val is None:
            raise KeyError(osm_id)
        return val

    def __len__(self) -> int:
        return self.n

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_GRAPH_SINGLETON: Optional[Dict[str, Any]] = None
_MODEL_SINGLETON: Optional[Dict[str, Any]] = None
_SPATIAL_TREES_SINGLETON: Optional[Dict[str, Any]] = None

def get_shared_graph() -> Dict[str, Any]:
    """Load compact road graph ONCE into RAM and return shared array dictionary."""
    global _GRAPH_SINGLETON
    if _GRAPH_SINGLETON is not None:
        return _GRAPH_SINGLETON

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Compact road graph missing: {GRAPH_PATH}")

    raw = np.load(GRAPH_PATH)
    lat = np.ascontiguousarray(raw["lat"], dtype=np.float32)
    lon = np.ascontiguousarray(raw["lon"], dtype=np.float32)
    node_ids = np.ascontiguousarray(raw["node_ids"], dtype=np.int64)
    offsets = np.ascontiguousarray(raw["offsets"], dtype=np.int32)
    targets = np.ascontiguousarray(raw["targets"], dtype=np.int32)
    distances = np.ascontiguousarray(raw["distances"], dtype=np.float32)
    node_risk_mult = np.ascontiguousarray(raw["node_risk_mult"], dtype=np.float32)

    # Precompute u_nodes for vectorized edge mapping
    u_nodes = np.repeat(np.arange(len(lat), dtype=np.int32), np.diff(offsets))

    fast_index = FastNodeIndex(node_ids)

    _GRAPH_SINGLETON = {
        "lat": lat,
        "lon": lon,
        "node_ids": node_ids,
        "offsets": offsets,
        "targets": targets,
        "distances": distances,
        "node_risk_mult": node_risk_mult,
        "u_nodes": u_nodes,
        "osm_id_to_index": fast_index,
        "node_count": len(lat),
        "edge_count": len(targets),
    }
    return _GRAPH_SINGLETON


def get_shared_model_bundle() -> Dict[str, Any]:
    """Load XGBoost Model V2 & Metadata ONCE into RAM."""
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is not None:
        return _MODEL_SINGLETON

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(f"Model V2 or metadata artifact missing: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    _MODEL_SINGLETON = {
        "model": model,
        "metadata": meta,
        "feature_order": meta.get("feature_order", []),
        "class_names": meta.get("class_names", ["High", "Low", "Medium"]),
    }
    return _MODEL_SINGLETON


def get_shared_spatial_trees() -> Dict[str, Any]:
    """
    Build Spatial KDTrees using lightweight standard Python JSON parsing.
    Completely avoids loading GeoPandas / Fiona / GDAL stack.
    """
    global _SPATIAL_TREES_SINGLETON
    if _SPATIAL_TREES_SINGLETON is not None:
        return _SPATIAL_TREES_SINGLETON

    drain_tree = None
    pop_tree = None
    pop_totals = None
    pop_names = None
    infra_tree = None

    # 1. Drainage Spatial KDTree
    if DRAINS_PATH.exists():
        try:
            with open(DRAINS_PATH, "r", encoding="utf-8") as f:
                drn_data = json.load(f)
            drn_coords = []
            for feat in drn_data.get("features", []):
                geom = feat.get("geometry")
                if geom and geom.get("type") == "Point":
                    coords = geom.get("coordinates")
                    if coords and len(coords) >= 2:
                        drn_coords.append([coords[1], coords[0]])
            if drn_coords:
                drain_tree = cKDTree(np.array(drn_coords, dtype=np.float32))
        except Exception:
            drain_tree = None

    # 2. District Population Spatial KDTree
    if POP_PATH.exists():
        try:
            with open(POP_PATH, "r", encoding="utf-8") as f:
                pop_data = json.load(f)
            pop_coords = []
            totals = []
            names = []
            for feat in pop_data.get("features", []):
                props = feat.get("properties", {})
                geom = feat.get("geometry")
                if geom and geom.get("type") == "Point":
                    coords = geom.get("coordinates")
                    if coords and len(coords) >= 2:
                        pop_coords.append([coords[1], coords[0]])
                        totals.append(float(props.get("population_total", 1500000.0)))
                        names.append(str(props.get("district", "Delhi District")))
            if pop_coords:
                pop_tree = cKDTree(np.array(pop_coords, dtype=np.float32))
                pop_totals = np.array(totals, dtype=np.float32)
                pop_names = np.array(names)
        except Exception:
            pop_tree = None

    # 3. OSM Infrastructure Spatial KDTree
    if INFRA_PATH.exists():
        try:
            with open(INFRA_PATH, "r", encoding="utf-8") as f:
                infra_data = json.load(f)
            infra_coords = []
            for el in infra_data.get("elements", []):
                lat = el.get("lat") or el.get("center", {}).get("lat")
                lon = el.get("lon") or el.get("center", {}).get("lon")
                if lat is not None and lon is not None:
                    infra_coords.append([lat, lon])
            if infra_coords:
                infra_tree = cKDTree(np.array(infra_coords, dtype=np.float32))
        except Exception:
            infra_tree = None

    _SPATIAL_TREES_SINGLETON = {
        "drain_tree": drain_tree,
        "pop_tree": pop_tree,
        "pop_totals": pop_totals,
        "pop_names": pop_names,
        "infra_tree": infra_tree,
    }
    return _SPATIAL_TREES_SINGLETON
