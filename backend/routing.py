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
- Risk multipliers retained
"""

import heapq
import math
from pathlib import Path

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
    Memory-efficient A* router.

    Internal graph representation:
    - node_ids: original OSM IDs
    - lat/lon: coordinate arrays
    - offsets/targets/distances: CSR-style adjacency arrays
    - node_risk_mult: compact NumPy array

    Important:
    nearest_node() returns the compact ARRAY INDEX,
    not the original OSM node ID. This avoids maintaining
    a huge Python dictionary and reduces RAM usage.
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

          # Load precomputed spatial flood risk.
        # Risk was calculated offline during graph generation.
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
        """
        Find nearest graph node using cKDTree.

        Returns:
            compact array index,
            snap distance in metres
        """

        if self.node_count == 0:
            raise ValueError(
                "No road nodes available in graph."
            )

        _, index = self.kdtree.query(
            [lat, lon]
        )

        index = int(index)

        nearest_lat = float(self.lat[index])
        nearest_lon = float(self.lon[index])

        distance = haversine(
            lat,
            lon,
            nearest_lat,
            nearest_lon,
        )

        # IMPORTANT:
        # Return compact array index directly.
        # This avoids the large node_id_to_index dictionary.
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

    def route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        risk: str = "Low",
        **kwargs,
    ) -> dict:

        if risk not in BASE_RISK_MULTIPLIERS:
            raise ValueError(
                f"Invalid risk level: {risk}. "
                "Use Low, Medium or High."
            )

        # nearest_node now returns compact ARRAY INDICES.
        start, start_snap_distance = self.nearest_node(
            start_lat,
            start_lon,
        )

        target, end_snap_distance = self.nearest_node(
            end_lat,
            end_lon,
        )

        # Original OSM IDs are obtained only when needed.
        start_node_id = int(
            self.node_ids[start]
        )

        end_node_id = int(
            self.node_ids[target]
        )

        if (
            start_snap_distance > 10000
            or end_snap_distance > 10000
        ):
            return {
                "status": "error",
                "message": (
                    "Coordinates are outside "
                    "the supported domain."
                ),
            }

        sensitivity = RISK_SENSITIVITY[risk]
        base_multiplier = BASE_RISK_MULTIPLIERS[risk]

        if start == target:

            return {
                "status": "ok",
                "routing_mode": "flood_aware",
                "algorithm": "A*",
                "risk_mode": risk,
                "risk_level": risk,
                "risk_multiplier": base_multiplier,
                "risk_basis": "spatial_proxy",
                "distance_m": 0.0,
                "estimated_cost": 0.0,
                "routing_cost": 0.0,
                "origin_snap_distance_m": round(
                    start_snap_distance,
                    2,
                ),
                "destination_snap_distance_m": round(
                    end_snap_distance,
                    2,
                ),
                "start_node": start_node_id,
                "end_node": end_node_id,
                "nodes_in_route": 1,
                "route": [
                    {
                        "lat": start_lat,
                        "lon": start_lon,
                    },
                    {
                        "lat": end_lat,
                        "lon": end_lon,
                    },
                ],
                "coordinates": [
                    [start_lat, start_lon],
                    [end_lat, end_lon],
                ],
            }

        # A* queue
        open_set = [
            (
                self.heuristic(start, target),
                0.0,
                start,
            )
        ]

        came_from = {}
        g_score = {
            start: 0.0
        }

        visited = set()

        found = False

        while open_set:

            _, current_g, current = (
                heapq.heappop(open_set)
            )

            if current in visited:
                continue

            visited.add(current)

            if current == target:
                found = True
                break

            current_risk = float(
                self.node_risk_mult[current]
            )

            edge_start = int(
                self.offsets[current]
            )

            edge_end = int(
                self.offsets[current + 1]
            )

            for edge_index in range(
                edge_start,
                edge_end,
            ):

                neighbour = int(
                    self.targets[edge_index]
                )

                base_distance = float(
                    self.distances[edge_index]
                )

                neighbour_risk = float(
                    self.node_risk_mult[neighbour]
                )

                spatial_multiplier = max(
                    current_risk,
                    neighbour_risk,
                )

                edge_cost_multiplier = (
                    1.0
                    + (
                        spatial_multiplier - 1.0
                    )
                    * sensitivity
                )

                edge_cost = (
                    base_distance
                    * edge_cost_multiplier
                )

                tentative_g = (
                    current_g + edge_cost
                )

                if tentative_g < g_score.get(
                    neighbour,
                    float("inf"),
                ):

                    came_from[neighbour] = current

                    g_score[neighbour] = (
                        tentative_g
                    )

                    estimated_total = (
                        tentative_g
                        + self.heuristic(
                            neighbour,
                            target,
                        )
                    )

                    heapq.heappush(
                        open_set,
                        (
                            estimated_total,
                            tentative_g,
                            neighbour,
                        ),
                    )

        if not found:

            return {
                "status": "error",
                "message": (
                    "No route found between "
                    "the supplied coordinates."
                ),
            }

        # Reconstruct compact-index path
        path = [target]

        current = target

        while current != start:

            current = came_from[current]

            path.append(current)

        path.reverse()

        # Convert compact indices to coordinates
        route_list = [
            {
                "lat": float(self.lat[index]),
                "lon": float(self.lon[index]),
            }
            for index in path
        ]

        coordinates = [
            [
                float(self.lat[index]),
                float(self.lon[index]),
            ]
            for index in path
        ]

        # Physical route distance
        physical_distance = 0.0

        for i in range(
            len(path) - 1
        ):

            current = path[i]
            next_node = path[i + 1]

            edge_start = int(
                self.offsets[current]
            )

            edge_end = int(
                self.offsets[current + 1]
            )

            for edge_index in range(
                edge_start,
                edge_end,
            ):

                if int(
                    self.targets[edge_index]
                ) == next_node:

                    physical_distance += float(
                        self.distances[edge_index]
                    )

                    break

        return {
            "status": "ok",
            "routing_mode": "flood_aware",
            "algorithm": "A*",
            "risk_mode": risk,
            "risk_level": risk,
            "risk_multiplier": base_multiplier,
            "risk_basis": "spatial_proxy",
            "start_node": start_node_id,
            "end_node": end_node_id,
            "origin_snap_distance_m": round(
                start_snap_distance,
                2,
            ),
            "destination_snap_distance_m": round(
                end_snap_distance,
                2,
            ),
            "distance_m": round(
                physical_distance,
                2,
            ),
            "estimated_cost": round(
                g_score[target],
                2,
            ),
            "routing_cost": round(
                g_score[target],
                2,
            ),
            "nodes_in_route": len(path),
            "route": route_list,
            "coordinates": coordinates,
        }
