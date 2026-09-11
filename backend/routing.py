"""
AquaG Road Routing Engine

Memory-efficient deployment version.

Uses the complete Delhi road graph but stores nodes/edges in compact
NumPy arrays instead of millions of Python dictionaries.

Routing:
- A*
- All original road nodes and edges preserved
- One-way direction preserved
- DEM + drainage spatial proxy retained
- Dynamic street-level waterlogging flood risk integration (Phase 3C)
"""

import heapq
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]

# Compact deployment graph
GRAPH_PATH = (
    ROOT
    / "data"
    / "processed"
    / "delhi_road_graph_compact.npz"
)

DATA_RAW_ROOT = ROOT / "data" / "raw"
DEM_PATH = DATA_RAW_ROOT / "dem" / "delhi_aw3d30.tif"
DRAINS_DIR = DATA_RAW_ROOT / "drainage"


BASE_RISK_MULTIPLIERS = {
    "Low": 1.0,
    "Medium": 2.0,
    "High": 5.0,
}


RISK_SENSITIVITY = {
    "Low": 1.0,
    "Medium": 1.5,
    "High": 2.0,
}


def haversine(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
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

    return 2 * earth_radius * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )


class AquaGRouter:
    """
    Memory-efficient A* router with dynamic flood-aware routing capability.

    Internal graph representation:
    - node_ids: original OSM IDs
    - lat/lon: coordinate arrays
    - offsets/targets/distances: CSR-style adjacency arrays
    - node_risk_mult: compact NumPy array
    - osm_id_to_index: fast lookup dictionary from OSM Node ID to array index
    """

    def __init__(self, graph_path=GRAPH_PATH):
        self.graph_path = Path(graph_path)

        if not self.graph_path.exists():
            raise FileNotFoundError(
                f"Compact road graph not found: {self.graph_path}"
            )

        print("Loading compact AquaG road graph...")

        graph = np.load(self.graph_path)

        # Original OSM node IDs
        self.node_ids = graph["node_ids"]

        # Fast O(1) mapping from OSM Node ID to compact array index
        self.osm_id_to_index = {
            int(nid): idx for idx, nid in enumerate(self.node_ids)
        }

        # Coordinates
        self.lat = graph["lat"]
        self.lon = graph["lon"]

        # CSR adjacency representation
        self.offsets = graph["offsets"]
        self.targets = graph["targets"]
        self.distances = graph["distances"]

        self.node_count = len(self.node_ids)
        self.edge_count = len(self.targets)

        print(
            f"Compact graph loaded: "
            f"{self.node_count:,} nodes, "
            f"{self.edge_count:,} edges"
        )

        # Coordinate array for nearest-node lookup
        node_coords = np.column_stack(
            (self.lat, self.lon)
        ).astype(np.float32, copy=False)

        self.node_coords = node_coords

        # cKDTree for O(log N) nearest-node snapping
        self.kdtree = cKDTree(node_coords)

        # Load precomputed spatial flood risk
        with np.load(GRAPH_PATH) as graph_data:
            self.node_risk_mult = graph_data["node_risk_mult"].astype(
                np.float32,
                copy=False,
            )

        if len(self.node_risk_mult) != self.node_count:
            raise ValueError(
                "Precomputed risk array size does not match road graph."
            )

        print("AquaGRouter initialization complete.")

    def nearest_node(
        self,
        lat: float,
        lon: float,
    ):
        """Find nearest graph node using cKDTree."""
        if self.node_count == 0:
            raise ValueError("No road nodes available in graph.")

        _, index = self.kdtree.query([lat, lon])
        index = int(index)

        nearest_lat = float(self.lat[index])
        nearest_lon = float(self.lon[index])

        distance = haversine(
            lat,
            lon,
            nearest_lat,
            nearest_lon,
        )

        return index, distance

    def heuristic(
        self,
        node_index: int,
        target_index: int,
    ) -> float:
        """A* straight-line distance heuristic."""
        return haversine(
            float(self.lat[node_index]),
            float(self.lon[node_index]),
            float(self.lat[target_index]),
            float(self.lon[target_index]),
        )

    def calculate_flood_penalty_multiplier(self, water_depth_cm: float) -> float:
        """
        Deterministic dynamic routing penalty multiplier based on water depth (cm).
        
        Thresholds:
        - depth < 10.0 cm: 1.0 (minimal/no penalty)
        - 10.0 <= depth < 25.0 cm: 1.5 to 2.5x (moderate penalty)
        - 25.0 <= depth < 50.0 cm: 3.0 to 6.0x (strong penalty)
        - 50.0 <= depth < 100.0 cm: 8.0 to 20.0x (very strong penalty)
        - depth >= 100.0 cm: 100.0+ (extreme penalty, avoid unless no alternative)
        """
        depth = max(0.0, float(water_depth_cm))
        if depth < 10.0:
            return 1.0
        elif depth < 25.0:
            return 1.5 + ((depth - 10.0) / 15.0) * 1.0
        elif depth < 50.0:
            return 3.0 + ((depth - 25.0) / 25.0) * 3.0
        elif depth < 100.0:
            return 8.0 + ((depth - 50.0) / 50.0) * 12.0
        else:
            return 100.0 + (depth - 100.0) * 2.0

    def route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        risk: str = "Low",
        scenario: str = "NORMAL",
        timestep: str = "T+0",
        rainfall_1h: Optional[float] = None,
        rainfall_3h: Optional[float] = None,
        rainfall_6h: Optional[float] = None,
        recent_rainfall_intensity: Optional[float] = None,
        flood_aware: bool = True,
    ) -> dict:

        if risk not in BASE_RISK_MULTIPLIERS:
            raise ValueError(
                f"Invalid risk level: {risk}. Use Low, Medium or High."
            )

        start, start_snap_distance = self.nearest_node(start_lat, start_lon)
        target, end_snap_distance = self.nearest_node(end_lat, end_lon)

        start_node_id = int(self.node_ids[start])
        end_node_id = int(self.node_ids[target])

        if start_snap_distance > 10000 or end_snap_distance > 10000:
            return {
                "status": "error",
                "message": "Coordinates are outside the supported domain.",
            }

        sensitivity = RISK_SENSITIVITY[risk]
        base_multiplier = BASE_RISK_MULTIPLIERS[risk]
        timestep_clean = str(timestep).upper().replace(" ", "+").strip()
        scenario_clean = str(scenario).upper().strip()

        if start == target:
            return {
                "status": "ok",
                "routing_mode": "flood_aware" if flood_aware else "static_risk",
                "algorithm": "A*",
                "risk_mode": risk,
                "risk_level": risk,
                "risk_multiplier": base_multiplier,
                "risk_basis": "model_derived_flood_aware_routing" if flood_aware else "spatial_proxy",
                "distance_m": 0.0,
                "physical_distance_m": 0.0,
                "estimated_cost": 0.0,
                "routing_cost": 0.0,
                "origin_snap_distance_m": round(start_snap_distance, 2),
                "destination_snap_distance_m": round(end_snap_distance, 2),
                "start_node": start_node_id,
                "end_node": end_node_id,
                "nodes_in_route": 1,
                "route": [{"lat": start_lat, "lon": start_lon}, {"lat": end_lat, "lon": end_lon}],
                "coordinates": [[start_lat, start_lon], [end_lat, end_lon]],
                "flood_aware": flood_aware,
                "scenario": scenario_clean,
                "timestep": timestep_clean,
                "route_risk_level": "Low",
                "flooded_segments_on_route": 0,
                "maximum_water_depth_cm": 0.0,
                "avoided_high_risk_segments": 0,
                "basis": "model_derived_flood_aware_routing" if flood_aware else "spatial_proxy",
            }

        # -------------------------------------------------------------------
        # Dynamic Corridor Flood Risk Mapping (Phase 3C)
        # -------------------------------------------------------------------
        dynamic_edge_penalties: Dict[Tuple[int, int], float] = {}
        dynamic_segment_depths: Dict[Tuple[int, int], float] = {}

        if flood_aware:
            min_lat = min(start_lat, end_lat)
            max_lat = max(start_lat, end_lat)
            min_lon = min(start_lon, end_lon)
            max_lon = max(start_lon, end_lon)

            lat_buf = max(0.06, (max_lat - min_lat) * 0.4)
            lon_buf = max(0.06, (max_lon - min_lon) * 0.4)

            corridor_bbox = [
                max(76.8, min_lon - lon_buf),
                max(28.3, min_lat - lat_buf),
                min(77.4, max_lon + lon_buf),
                min(28.9, max_lat + lat_buf),
            ]

            if rainfall_1h is None:
                if scenario_clean == "EXTREME":
                    r1h, r3h, r6h, rint = 110.0, 180.0, 250.0, 55.0
                elif scenario_clean == "HEAVY":
                    r1h, r3h, r6h, rint = 75.0, 130.0, 180.0, 37.5
                elif scenario_clean == "MODERATE":
                    r1h, r3h, r6h, rint = 45.0, 85.0, 130.0, 22.5
                else:
                    r1h, r3h, r6h, rint = 10.0, 20.0, 30.0, 5.0
            else:
                r1h = rainfall_1h
                r3h = rainfall_3h or (r1h * 2.0)
                r6h = rainfall_6h or (r1h * 3.0)
                rint = recent_rainfall_intensity or (r1h * 0.5)

            try:
                from waterlogging import get_street_waterlogging_geojson
                wl_geojson = get_street_waterlogging_geojson(
                    scenario=scenario_clean,
                    timestep=timestep_clean,
                    rainfall_1h=r1h,
                    rainfall_3h=r3h,
                    rainfall_6h=r6h,
                    recent_rainfall_intensity=rint,
                    bbox=corridor_bbox,
                    max_segments=5000,
                )
                features = wl_geojson.get("features", [])
                for f in features:
                    props = f.get("properties", {})
                    u_osm = props.get("start_node")
                    v_osm = props.get("end_node")
                    if u_osm is None or v_osm is None:
                        continue

                    u_idx = self.osm_id_to_index.get(int(u_osm))
                    v_idx = self.osm_id_to_index.get(int(v_osm))
                    if u_idx is None or v_idx is None:
                        continue

                    depth_cm = float(props.get("water_depth_cm", 0.0))
                    pen_mult = self.calculate_flood_penalty_multiplier(depth_cm)

                    dynamic_segment_depths[(u_idx, v_idx)] = depth_cm
                    dynamic_segment_depths[(v_idx, u_idx)] = depth_cm
                    dynamic_edge_penalties[(u_idx, v_idx)] = pen_mult
                    dynamic_edge_penalties[(v_idx, u_idx)] = pen_mult
            except Exception as e:
                print(f"Warning: Dynamic waterlogging query failed non-blocking: {e}")

        # A* Search
        open_set = [(self.heuristic(start, target), 0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        visited = set()
        found = False

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current in visited:
                continue

            visited.add(current)

            if current == target:
                found = True
                break

            current_risk = float(self.node_risk_mult[current])
            edge_start = int(self.offsets[current])
            edge_end = int(self.offsets[current + 1])

            for edge_index in range(edge_start, edge_end):
                neighbour = int(self.targets[edge_index])
                base_distance = float(self.distances[edge_index])
                neighbour_risk = float(self.node_risk_mult[neighbour])

                spatial_multiplier = max(current_risk, neighbour_risk)
                dyn_penalty = dynamic_edge_penalties.get((current, neighbour), 1.0)
                total_risk_multiplier = max(spatial_multiplier, dyn_penalty)

                edge_cost_multiplier = 1.0 + (total_risk_multiplier - 1.0) * sensitivity
                edge_cost = base_distance * edge_cost_multiplier

                tentative_g = current_g + edge_cost

                if tentative_g < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    estimated_total = tentative_g + self.heuristic(neighbour, target)
                    heapq.heappush(open_set, (estimated_total, tentative_g, neighbour))

        if not found:
            return {
                "status": "error",
                "message": "No route found between the supplied coordinates.",
            }

        # Reconstruct path
        path = [target]
        current = target
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()

        route_list = [
            {"lat": float(self.lat[index]), "lon": float(self.lon[index])}
            for index in path
        ]
        coordinates = [
            [float(self.lat[index]), float(self.lon[index])]
            for index in path
        ]

        # Calculate physical distance & dynamic route metrics
        physical_distance = 0.0
        flooded_segments_on_route = 0
        maximum_water_depth_cm = 0.0
        route_edge_set = set()

        for i in range(len(path) - 1):
            curr_idx = path[i]
            next_idx = path[i + 1]
            route_edge_set.add((curr_idx, next_idx))

            depth_val = max(
                dynamic_segment_depths.get((curr_idx, next_idx), 0.0),
                dynamic_segment_depths.get((next_idx, curr_idx), 0.0)
            )
            if depth_val >= 10.0:
                flooded_segments_on_route += 1
            if depth_val > maximum_water_depth_cm:
                maximum_water_depth_cm = depth_val

            edge_start = int(self.offsets[curr_idx])
            edge_end = int(self.offsets[curr_idx + 1])
            for edge_index in range(edge_start, edge_end):
                if int(self.targets[edge_index]) == next_idx:
                    physical_distance += float(self.distances[edge_index])
                    break

        # Calculate count of high-risk corridor segments avoided
        avoided_count = 0
        visited_corridor_pairs = set()
        for (u, v), d_val in dynamic_segment_depths.items():
            pair_key = tuple(sorted([u, v]))
            if pair_key in visited_corridor_pairs:
                continue
            visited_corridor_pairs.add(pair_key)

            if d_val >= 25.0:
                if (u, v) not in route_edge_set and (v, u) not in route_edge_set:
                    avoided_count += 1

        if maximum_water_depth_cm >= 50.0:
            route_risk_level = "Critical"
        elif maximum_water_depth_cm >= 25.0:
            route_risk_level = "High"
        elif maximum_water_depth_cm >= 10.0:
            route_risk_level = "Medium"
        else:
            route_risk_level = "Low"

        return {
            "status": "ok",
            "routing_mode": "flood_aware" if flood_aware else "static_risk",
            "algorithm": "A*",
            "risk_mode": risk,
            "risk_level": risk,
            "risk_multiplier": base_multiplier,
            "risk_basis": "spatial_proxy",
            "start_node": start_node_id,
            "end_node": end_node_id,
            "origin_snap_distance_m": round(start_snap_distance, 2),
            "destination_snap_distance_m": round(end_snap_distance, 2),
            "distance_m": round(physical_distance, 2),
            "physical_distance_m": round(physical_distance, 2),
            "estimated_cost": round(g_score[target], 2),
            "routing_cost": round(g_score[target], 2),
            "nodes_in_route": len(path),
            "route": route_list,
            "coordinates": coordinates,
            "flood_aware": flood_aware,
            "scenario": scenario_clean,
            "timestep": timestep_clean,
            "route_risk_level": route_risk_level,
            "flooded_segments_on_route": flooded_segments_on_route,
            "maximum_water_depth_cm": round(maximum_water_depth_cm, 1),
            "avoided_high_risk_segments": avoided_count,
            "basis": "model_derived_flood_aware_routing" if flood_aware else "spatial_proxy",
        }
