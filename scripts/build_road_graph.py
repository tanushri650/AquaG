import json
import math
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "raw" / "roads" / "delhi_road_network.json"
OUTPUT_FILE = ROOT / "data" / "processed" / "delhi_road_graph.json"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Keep only routable road types.
ALLOWED_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
}

# Flood-risk multipliers.
# These are deterministic routing costs, NOT flood-depth predictions.
FLOOD_MULTIPLIERS = {
    "Low": 1.0,
    "Medium": 2.0,
    "High": 5.0,
}


# ---------------------------------------------------------
# Distance calculation
# ---------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two geographic coordinates.
    Returns metres.
    """

    R = 6371000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


def geometry_distance(geometry):
    """
    Calculate total length of an OSM way geometry.
    """

    total = 0.0

    for i in range(len(geometry) - 1):
        p1 = geometry[i]
        p2 = geometry[i + 1]

        total += haversine_m(
            p1["lat"],
            p1["lon"],
            p2["lat"],
            p2["lon"],
        )

    return total


# ---------------------------------------------------------
# Main graph construction
# ---------------------------------------------------------

def build_graph():

    print("Loading OSM road data...")
    print(f"Input: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        osm = json.load(f)

    elements = osm.get("elements", [])

    print(f"Total OSM elements: {len(elements):,}")

    nodes = {}
    ways = []

    # -----------------------------------------------------
    # First pass: collect OSM nodes and road ways
    # -----------------------------------------------------

    for element in elements:

        element_type = element.get("type")

        if element_type == "node":

            node_id = element.get("id")

            lat = element.get("lat")
            lon = element.get("lon")

            if node_id is not None and lat is not None and lon is not None:
                nodes[node_id] = {
                    "lat": lat,
                    "lon": lon,
                }

        elif element_type == "way":

            tags = element.get("tags", {})
            highway = tags.get("highway")

            if highway not in ALLOWED_HIGHWAYS:
                continue

            geometry = element.get("geometry", [])

            if len(geometry) < 2:
                continue

            ways.append({
                "id": element.get("id"),
                "nodes": element.get("nodes", []),
                "geometry": geometry,
                "highway": highway,
                "name": tags.get("name"),
                "oneway": tags.get("oneway"),
            })

    print(f"OSM nodes collected: {len(nodes):,}")
    print(f"Routable road ways: {len(ways):,}")

    # -----------------------------------------------------
    # Graph structures
    # -----------------------------------------------------

    graph_nodes = {}
    graph_edges = []

    # Used to assign compact integer IDs to graph nodes.
    node_seen = set()

    def add_graph_node(node_id, lat, lon):

        if node_id not in graph_nodes:
            graph_nodes[node_id] = {
                "id": node_id,
                "lat": lat,
                "lon": lon,
            }

    # -----------------------------------------------------
    # Build edges
    # -----------------------------------------------------

    for index, way in enumerate(ways):

        geometry = way["geometry"]
        osm_nodes = way["nodes"]

        # Some OSM ways may not have node IDs.
        # In that case, create deterministic geometry IDs.
        if len(osm_nodes) == len(geometry):

            point_ids = osm_nodes

        else:

            point_ids = []

            for point_index, point in enumerate(geometry):

                point_ids.append(
                    f"way_{way['id']}_{point_index}"
                )

        # Add graph nodes.
        for point_id, point in zip(point_ids, geometry):

            add_graph_node(
                point_id,
                point["lat"],
                point["lon"],
            )

        # Create consecutive edges.
        for i in range(len(point_ids) - 1):

            source = point_ids[i]
            target = point_ids[i + 1]

            p1 = geometry[i]
            p2 = geometry[i + 1]

            distance = haversine_m(
                p1["lat"],
                p1["lon"],
                p2["lat"],
                p2["lon"],
            )

            if distance <= 0:
                continue

            edge = {
                "source": source,
                "target": target,
                "distance_m": round(distance, 2),
                "road_type": way["highway"],
                "road_name": way["name"],
            }

            graph_edges.append(edge)

            # -------------------------------------------------
            # Direction handling
            # -------------------------------------------------

            oneway = str(way.get("oneway", "")).lower()

            if oneway not in {"yes", "1", "true"}:

                reverse_edge = {
                    "source": target,
                    "target": source,
                    "distance_m": round(distance, 2),
                    "road_type": way["highway"],
                    "road_name": way["name"],
                }

                graph_edges.append(reverse_edge)

        if (index + 1) % 10000 == 0:

            print(
                f"Processed {index + 1:,}/{len(ways):,} ways..."
            )

    # -----------------------------------------------------
    # Remove nodes that never participate in edges
    # -----------------------------------------------------

    used_nodes = set()

    for edge in graph_edges:
        used_nodes.add(edge["source"])
        used_nodes.add(edge["target"])

    graph_nodes = {
        node_id: node
        for node_id, node in graph_nodes.items()
        if node_id in used_nodes
    }

    # -----------------------------------------------------
    # Save compact graph
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph = {
        "metadata": {
            "source": "Delhi OSM road network",
            "routing_algorithm": "A*",
            "distance": "Haversine metres",
            "risk_multipliers": FLOOD_MULTIPLIERS,
            "note": (
                "Risk multipliers are deterministic routing "
                "costs and are not observed flood depths."
            ),
        },
        "nodes": list(graph_nodes.values()),
        "edges": graph_edges,
    }

    print("Writing compact graph...")
    print(f"Output: {OUTPUT_FILE}")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            graph,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    print()
    print("=" * 60)
    print("ROAD GRAPH BUILD COMPLETE")
    print("=" * 60)
    print(f"Graph nodes : {len(graph_nodes):,}")
    print(f"Graph edges : {len(graph_edges):,}")
    print(f"Output file : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    build_graph()