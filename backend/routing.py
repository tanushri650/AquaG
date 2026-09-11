"""
AquaG Road Routing Engine
Stage 6C: Authoritative Spatial Flood-Aware Routing Engine on the Preprocessed Delhi OSM Road Graph.

Documented Basis:
- MODEL-DERIVED: AquaG XGBoost V2 flood severity (zone-level classification).
- SPATIAL PROXY: Prototype spatial proxy thresholds derived from DEM elevation (delhi_aw3d30.tif) 
  and drainage network proximity (delhi_drains_mpd1976_full.geojson). 
  These are deterministic routing penalties, NOT measured street-level water depths.
- ROUTING COST: Deterministic multiplier M(edge) applied to individual road segments in A*.
- OBSERVED DATA: No street-level flood depth ground truth currently available.
"""

import heapq
import json
import math
from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "processed" / "delhi_road_graph.json"
DATA_RAW_ROOT = ROOT / "data" / "raw"
DEM_PATH = DATA_RAW_ROOT / "dem" / "delhi_aw3d30.tif"
DRAINS_DIR = DATA_RAW_ROOT / "drainage"

# Base deterministic spatial risk multipliers
BASE_RISK_MULTIPLIERS = {
    "Low": 1.0,
    "Medium": 2.0,
    "High": 5.0,
}

# Sensitivity scaling per requested risk level (routing avoidance strength)
RISK_SENSITIVITY = {
    "Low": 1.0,
    "Medium": 1.5,
    "High": 2.0,
}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return approximate distance between two coordinates in metres."""
    earth_radius = 6_371_000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(d_lambda / 2) ** 2
    )

    return 2 * earth_radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class AquaGRouter:
    """A* router using spatially varying edge risk costs precomputed from DEM & drainage layers."""

    def __init__(self, graph_path=GRAPH_PATH):
        self.graph_path = Path(graph_path)

        if not self.graph_path.exists():
            raise FileNotFoundError(f"Road graph file not found: {self.graph_path}")

        with open(self.graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)

        self.metadata = graph.get("metadata", {})
        self.nodes = {
            int(node["id"]): node
            for node in graph["nodes"]
        }

        self.adjacency = {}
        for edge in graph["edges"]:
            source = int(edge["source"])
            self.adjacency.setdefault(source, []).append(edge)

        self.node_id_list = list(self.nodes.keys())
        node_coords = np.array([
            [self.nodes[nid]["lat"], self.nodes[nid]["lon"]]
            for nid in self.node_id_list
        ], dtype=np.float64)
        
        # cKDTree for O(log N) nearest-node snapping
        self.kdtree = cKDTree(node_coords)

        # Precompute spatial flood risk for all road nodes
        self._precompute_spatial_risk(node_coords)

    def _precompute_spatial_risk(self, node_coords: np.ndarray):
        """Precompute node-level spatial risk levels and multipliers once during initialization."""
        self.node_risk_level = {}
        self.node_risk_mult = {}

        # 1. Load DEM dataset
        if not DEM_PATH.exists():
            # Fallback to default Low risk if DEM missing
            for nid in self.node_id_list:
                self.node_risk_level[nid] = "Low"
                self.node_risk_mult[nid] = 1.0
            return

        with rasterio.open(DEM_PATH) as dem_src:
            dem_data = dem_src.read(1)
            inv_transform = ~dem_src.transform
            lons = node_coords[:, 1]
            lats = node_coords[:, 0]
            cols, rows = inv_transform * (lons, lats)
            rows = np.clip(rows.astype(int), 0, dem_data.shape[0] - 1)
            cols = np.clip(cols.astype(int), 0, dem_data.shape[1] - 1)
            elevations = dem_data[rows, cols]

        # 2. Load Drainage GeoJSON and build drainage cKDTree
        drn_files = [
            DRAINS_DIR / "delhi_drains_mpd1976_full.geojson",
            DRAINS_DIR / "delhi_untraceable_drains_mpd1976-1.geojson",
        ]
        drain_pts = []
        for p in drn_files:
            if p.exists():
                try:
                    gdf = gpd.read_file(p)
                    for geom in gdf.geometry:
                        if geom is not None and not geom.is_empty:
                            if geom.geom_type == "Point":
                                drain_pts.append((geom.y, geom.x))
                            elif geom.geom_type in ("LineString", "LinearRing"):
                                drain_pts.extend([(pt[1], pt[0]) for pt in geom.coords])
                            elif geom.geom_type == "MultiLineString":
                                for ls in geom.geoms:
                                    drain_pts.extend([(pt[1], pt[0]) for pt in ls.coords])
                except Exception:
                    pass

        if drain_pts:
            drain_pts_arr = np.array(drain_pts, dtype=np.float64)
            drain_tree = cKDTree(drain_pts_arr)
            dist_deg, _ = drain_tree.query(node_coords)
            dist_m = dist_deg * 111_000.0
        else:
            dist_m = np.full(len(node_coords), 1000.0)

        # 3. Classify node spatial risk using prototype proxy thresholds:
        # High: elevation < 210m AND drain distance < 50m  => 5.0
        # Medium: elevation < 215m AND drain distance < 200m => 2.0
        # Low: otherwise => 1.0
        high_mask = (elevations < 210.0) & (dist_m < 50.0)
        med_mask = (elevations < 215.0) & (dist_m < 200.0) & (~high_mask)

        for i, nid in enumerate(self.node_id_list):
            if high_mask[i]:
                self.node_risk_level[nid] = "High"
                self.node_risk_mult[nid] = 5.0
            elif med_mask[i]:
                self.node_risk_level[nid] = "Medium"
                self.node_risk_mult[nid] = 2.0
            else:
                self.node_risk_level[nid] = "Low"
                self.node_risk_mult[nid] = 1.0

    def nearest_node(self, lat: float, lon: float):
        """Find the closest graph node to a coordinate in O(log N) using cKDTree."""
        if not self.node_id_list:
            raise ValueError("No road nodes available in graph.")

        _, idx = self.kdtree.query([lat, lon])
        nearest_id = self.node_id_list[idx]
        node = self.nodes[nearest_id]
        distance = haversine(lat, lon, node["lat"], node["lon"])
        return nearest_id, distance

    def heuristic(self, node_id: int, target_id: int) -> float:
        """A* heuristic: straight-line distance to target."""
        node = self.nodes[node_id]
        target = self.nodes[target_id]
        return haversine(node["lat"], node["lon"], target["lat"], target["lon"])

    def route(self, start_lat: float, start_lon: float, end_lat: float, end_lon: float, risk: str = "Low") -> dict:
        """Calculate a route using A* with spatially varying edge risk costs."""
        if risk not in BASE_RISK_MULTIPLIERS:
            raise ValueError(
                f"Invalid risk level: {risk}. Use Low, Medium or High."
            )

        start_node, start_snap_distance = self.nearest_node(start_lat, start_lon)
        end_node, end_snap_distance = self.nearest_node(end_lat, end_lon)

        # Out-of-domain validation (> 10km snap distance)
        if start_snap_distance > 10000 or end_snap_distance > 10000:
            return {
                "status": "error",
                "message": "Coordinates are outside the supported domain.",
            }

        sensitivity = RISK_SENSITIVITY.get(risk, 1.0)
        base_multiplier = BASE_RISK_MULTIPLIERS[risk]

        if start_node == end_node:
            return {
                "status": "ok",
                "routing_mode": "flood_aware",
                "risk_mode": risk,
                "risk_level": risk,
                "risk_multiplier": base_multiplier,
                "risk_basis": "spatial_proxy",
                "distance_m": 0.0,
                "estimated_cost": 0.0,
                "routing_cost": 0.0,
                "origin_snap_distance_m": round(start_snap_distance, 2),
                "destination_snap_distance_m": round(end_snap_distance, 2),
                "start_node": start_node,
                "end_node": end_node,
                "nodes_in_route": 1,
                "route": [{"lat": start_lat, "lon": start_lon}, {"lat": end_lat, "lon": end_lon}],
                "coordinates": [[start_lat, start_lon], [end_lat, end_lon]],
            }

        # A* priority queue: (estimated_total_cost, tentative_g, current_node_id)
        open_set = [(self.heuristic(start_node, end_node), 0.0, start_node)]
        came_from = {}
        g_score = {start_node: 0.0}
        visited = set()

        found = False

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current in visited:
                continue

            visited.add(current)

            if current == end_node:
                found = True
                break

            current_risk_mult = self.node_risk_mult.get(current, 1.0)

            for edge in self.adjacency.get(current, []):
                neighbour = int(edge["target"])
                base_distance = float(edge["distance_m"])

                # Spatially varying edge risk multiplier: M(edge) = max(M(source), M(target))
                neighbour_risk_mult = self.node_risk_mult.get(neighbour, 1.0)
                edge_spatial_mult = max(current_risk_mult, neighbour_risk_mult)

                # Apply risk sensitivity scaling
                edge_cost_multiplier = 1.0 + (edge_spatial_mult - 1.0) * sensitivity
                edge_cost = base_distance * edge_cost_multiplier
                tentative_g = current_g + edge_cost

                if tentative_g < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    estimated_total = tentative_g + self.heuristic(neighbour, end_node)
                    heapq.heappush(open_set, (estimated_total, tentative_g, neighbour))

        if not found:
            return {
                "status": "error",
                "message": "No route found between the supplied coordinates.",
            }

        # Reconstruct path
        path = [end_node]
        current = end_node
        while current != start_node:
            current = came_from[current]
            path.append(current)
        path.reverse()

        route_list = [
            {"lat": self.nodes[node_id]["lat"], "lon": self.nodes[node_id]["lon"]}
            for node_id in path
        ]
        coordinates = [
            [self.nodes[node_id]["lat"], self.nodes[node_id]["lon"]]
            for node_id in path
        ]

        # Physical route distance calculation
        physical_distance = 0.0
        for i in range(len(path) - 1):
            curr_n = path[i]
            next_n = path[i + 1]
            for edge in self.adjacency.get(curr_n, []):
                if int(edge["target"]) == next_n:
                    physical_distance += float(edge["distance_m"])
                    break

        return {
            "status": "ok",
            "routing_mode": "flood_aware",
            "algorithm": "A*",
            "risk_mode": risk,
            "risk_level": risk,
            "risk_multiplier": base_multiplier,
            "risk_basis": "spatial_proxy",
            "start_node": start_node,
            "end_node": end_node,
            "origin_snap_distance_m": round(start_snap_distance, 2),
            "destination_snap_distance_m": round(end_snap_distance, 2),
            "distance_m": round(physical_distance, 2),
            "estimated_cost": round(g_score[end_node], 2),
            "routing_cost": round(g_score[end_node], 2),
            "nodes_in_route": len(path),
            "route": route_list,
            "coordinates": coordinates,
        }