"""
Phase 3A Backend Tests
Verifies /infrastructure, /pumps, and /drainage endpoints, geometry integrity, bbox filtering,
and coordinate non-fabrication rules.
"""

import sys
import os
import pathlib
from fastapi.testclient import TestClient

# Path resolution for imports
backend_dir = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir))

from main import app

client = TestClient(app)

DELHI_BBOX_STR = "77.15,28.55,77.25,28.65"


def test_1_infrastructure_valid_geojson():
    response = client.get("/infrastructure")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    assert "metadata" in data
    assert data["metadata"]["source"] == "OSM"


def test_2_infrastructure_bbox_filtering():
    response = client.get(f"/infrastructure?bbox={DELHI_BBOX_STR}")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0
    assert len(data["features"]) <= 500  # Default cap

    min_lon, min_lat, max_lon, max_lat = [float(x) for x in DELHI_BBOX_STR.split(",")]
    for feature in data["features"]:
        lon, lat = feature["geometry"]["coordinates"]
        assert min_lon <= lon <= max_lon
        assert min_lat <= lat <= max_lat


def test_3_infrastructure_properties_presence():
    response = client.get(f"/infrastructure?bbox={DELHI_BBOX_STR}")
    assert response.status_code == 200
    data = response.json()
    assert len(data["features"]) > 0
    
    first_prop = data["features"][0]["properties"]
    assert "name" in first_prop
    assert "category" in first_prop
    assert "critical" in first_prop
    assert "source" in first_prop
    assert first_prop["source"] == "OSM"
    assert isinstance(first_prop["critical"], bool)


def test_4_drainage_valid_geojson():
    response = client.get("/drainage")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    assert "metadata" in data
    assert data["metadata"]["source"] == "MPD-1976"


def test_5_drainage_bbox_filtering():
    response = client.get(f"/drainage?bbox={DELHI_BBOX_STR}")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0

    min_lon, min_lat, max_lon, max_lat = [float(x) for x in DELHI_BBOX_STR.split(",")]
    for feature in data["features"]:
        lon, lat = feature["geometry"]["coordinates"]
        assert min_lon <= lon <= max_lon
        assert min_lat <= lat <= max_lat


def test_6_drainage_geometry_matches_source_point():
    response = client.get("/drainage")
    assert response.status_code == 200
    data = response.json()
    assert len(data["features"]) > 0
    
    for feature in data["features"]:
        assert feature["geometry"]["type"] == "Point"
        assert len(feature["geometry"]["coordinates"]) == 2
        prop = feature["properties"]
        assert "drain_name" in prop
        assert "basin" in prop
        assert "status" in prop
        assert "source" in prop


def test_7_pumps_source_derived_metadata():
    response = client.get("/pumps")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["total_stations"] == 10
    assert "Delhi Flood Control Order 2025" in data["source"]
    assert isinstance(data["stations"], list)
    assert len(data["stations"]) == 10


def test_8_pumps_coordinates_not_fabricated():
    response = client.get("/pumps")
    assert response.status_code == 200
    data = response.json()
    assert data["coordinates_available"] is False
    assert data["telemetry_available"] is False

    for station in data["stations"]:
        assert station["coordinates_available"] is False
        assert station["latitude"] is None
        assert station["longitude"] is None
        assert station["telemetry_available"] is False
