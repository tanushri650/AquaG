"""
Unit tests for AquaG Street-Level Waterlogging Engine (Stage 12 Phase 1).
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure backend modules are importable
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from main import app

client = TestClient(app)

SMALL_DELHI_BBOX = [77.19, 28.60, 77.23, 28.64]


def test_bbox_validation():
    """Test rejection of invalid bounding box arrays."""
    # Bad length
    res = client.post(
        "/waterlogging",
        json={
            "scenario": "NORMAL",
            "timestep": "T+0",
            "rainfall_1h": 10.0,
            "rainfall_3h": 20.0,
            "rainfall_6h": 30.0,
            "recent_rainfall_intensity": 5.0,
            "bbox": [77.19, 28.60, 77.23],
        },
    )
    assert res.status_code == 422

    # Min >= Max
    res = client.post(
        "/waterlogging",
        json={
            "scenario": "NORMAL",
            "timestep": "T+0",
            "rainfall_1h": 10.0,
            "rainfall_3h": 20.0,
            "rainfall_6h": 30.0,
            "recent_rainfall_intensity": 5.0,
            "bbox": [77.25, 28.60, 77.15, 28.64],
        },
    )
    assert res.status_code == 422


def test_geojson_validity_and_schema():
    """Test GeoJSON FeatureCollection structure and properties."""
    payload = {
        "scenario": "MODERATE",
        "timestep": "T+1",
        "rainfall_1h": 45.0,
        "rainfall_3h": 85.0,
        "rainfall_6h": 130.0,
        "recent_rainfall_intensity": 22.5,
        "bbox": SMALL_DELHI_BBOX,
    }
    res = client.post("/waterlogging", json=payload)
    assert res.status_code == 200, f"Waterlogging API failed: {res.text}"

    data = res.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    assert "metadata" in data
    assert isinstance(data["features"], list)

    features = data["features"]
    assert len(features) > 0, "Should return segments for small Delhi bbox"
    assert len(features) <= 2500, "Should cap segment count for performance"

    # Inspect first feature
    feat = features[0]
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "LineString"
    coords = feat["geometry"]["coordinates"]
    assert len(coords) == 2
    assert len(coords[0]) == 2
    assert len(coords[1]) == 2

    props = feat["properties"]
    assert props["water_depth_cm"] >= 0.0, "Depth must be non-negative"
    assert props["severity"] in ("Low", "Medium", "High")
    assert 0.0 <= props["high_probability"] <= 1.0
    assert 0.0 <= props["medium_probability"] <= 1.0
    assert 0.0 <= props["low_probability"] <= 1.0
    assert props["basis"] == "hydro_spatial_proxy"


def test_rainfall_scenario_and_timestep_effects():
    """Test that higher rainfall and progression timesteps increase depth proxies."""
    bbox = SMALL_DELHI_BBOX

    # 1. Normal T+0
    res_norm = client.post(
        "/waterlogging",
        json={
            "scenario": "NORMAL",
            "timestep": "T+0",
            "rainfall_1h": 10.0,
            "rainfall_3h": 20.0,
            "rainfall_6h": 30.0,
            "recent_rainfall_intensity": 5.0,
            "bbox": bbox,
        },
    )
    assert res_norm.status_code == 200
    feats_norm = res_norm.json()["features"]

    # 2. Heavy T+2
    res_heavy = client.post(
        "/waterlogging",
        json={
            "scenario": "HEAVY",
            "timestep": "T+2",
            "rainfall_1h": 75.0,
            "rainfall_3h": 130.0,
            "rainfall_6h": 180.0,
            "recent_rainfall_intensity": 37.5,
            "bbox": bbox,
        },
    )
    assert res_heavy.status_code == 200
    feats_heavy = res_heavy.json()["features"]

    avg_depth_norm = sum(f["properties"]["water_depth_cm"] for f in feats_norm) / len(feats_norm)
    avg_depth_heavy = sum(f["properties"]["water_depth_cm"] for f in feats_heavy) / len(feats_heavy)

    assert avg_depth_heavy > avg_depth_norm, (
        f"HEAVY T+2 avg depth ({avg_depth_heavy:.1f}cm) must exceed NORMAL T+0 avg depth ({avg_depth_norm:.1f}cm)"
    )


def test_existing_endpoints_regression():
    """Verify that adding /waterlogging did not break existing API endpoints."""
    assert client.get("/health").status_code == 200
    assert client.get("/zones").status_code == 200
    assert client.post("/flood_info", json={"lat": 28.6139, "lon": 77.2090}).status_code == 200
    assert client.post(
        "/predict",
        json={
            "rainfall_1h": 45.0,
            "rainfall_3h": 85.0,
            "rainfall_6h": 130.0,
            "recent_rainfall_intensity": 22.5,
            "elevation": 208.5,
            "slope": 1.8,
            "distance_to_drain": 45.0,
            "distance_to_road": 20.0,
            "distance_to_infra": 150.0,
            "population_total": 120000,
            "critical_infra_flag": 1,
        },
    ).status_code == 200


def test_lats_lons_initialization_regression():
    """Ensure _LATS and _LONS are non-None NumPy arrays before spatial filtering."""
    import numpy as np
    import waterlogging
    waterlogging._init_waterlogging_resources()
    assert waterlogging._LATS is not None, "_LATS must be initialized"
    assert waterlogging._LONS is not None, "_LONS must be initialized"
    assert isinstance(waterlogging._LATS, np.ndarray), "_LATS must be a NumPy array"
    assert isinstance(waterlogging._LONS, np.ndarray), "_LONS must be a NumPy array"
    assert len(waterlogging._LATS) > 0, "_LATS array must not be empty"
    assert len(waterlogging._LONS) > 0, "_LONS array must not be empty"
