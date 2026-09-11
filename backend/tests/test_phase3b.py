"""
Phase 3B Backend Unit Tests
Verifies Population Priority Engine, Alerts & Triage Engine, dataset integrity,
deterministic scoring, feature caps, and scientific labeling rules.
"""

import sys
import os
import pathlib
import numpy as np
import geopandas as gpd
from fastapi.testclient import TestClient

# Path resolution for imports
backend_dir = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir))

from main import app
from population_priority import calculate_population_priority_score, POP_PATH

client = TestClient(app)

DELHI_BBOX_STR = "77.15,28.55,77.25,28.65"


def test_1_population_dataset_loads():
    assert POP_PATH.exists(), f"Population dataset missing: {POP_PATH}"
    gdf = gpd.read_file(POP_PATH)
    assert len(gdf) > 0


def test_2_population_field_identified():
    gdf = gpd.read_file(POP_PATH)
    assert "population_total" in gdf.columns
    assert "district" in gdf.columns
    assert np.issubdtype(gdf["population_total"].dtype, np.number)


def test_3_population_geometry_valid():
    gdf = gpd.read_file(POP_PATH)
    assert not gdf.geometry.is_empty.any()
    assert (gdf.geometry.type == "Point").all()


def test_4_population_bbox_filtering():
    res = client.get(f"/population-priority?scenario=NORMAL&timestep=T+0&bbox={DELHI_BBOX_STR}")
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0

    min_lon, min_lat, max_lon, max_lat = [float(x) for x in DELHI_BBOX_STR.split(",")]
    for feature in data["features"]:
        coords = feature["geometry"]["coordinates"]
        mid_lon = (coords[0][0] + coords[1][0]) / 2.0
        mid_lat = (coords[0][1] + coords[1][1]) / 2.0
        assert min_lon <= mid_lon <= max_lon
        assert min_lat <= mid_lat <= max_lat


def test_5_population_priority_endpoint_geojson():
    res = client.get(f"/population-priority?scenario=HEAVY&timestep=T+2&bbox={DELHI_BBOX_STR}")
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"
    assert data["metadata"]["basis"] == "district_population_exposure_proxy"
    assert data["metadata"]["scenario"] == "HEAVY"
    assert data["metadata"]["timestep"] == "T+2"


def test_6_priority_levels_valid():
    res = client.get(f"/population-priority?scenario=HEAVY&timestep=T+2&bbox={DELHI_BBOX_STR}")
    assert res.status_code == 200
    data = res.json()
    valid_levels = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    for f in data["features"]:
        level = f["properties"]["priority_level"]
        assert level in valid_levels


def test_7_priority_score_deterministic():
    s1, l1 = calculate_population_priority_score(
        water_depth_cm=85.0,
        severity="High",
        population_exposure=2500000.0,
        critical_infra_flag=1,
    )
    s2, l2 = calculate_population_priority_score(
        water_depth_cm=85.0,
        severity="High",
        population_exposure=2500000.0,
        critical_infra_flag=1,
    )
    assert s1 == s2
    assert l1 == l2 == "CRITICAL"
    assert 0.0 <= s1 <= 100.0


def test_8_critical_infra_increases_priority():
    score_normal, _ = calculate_population_priority_score(
        water_depth_cm=45.0,
        severity="Medium",
        population_exposure=1500000.0,
        critical_infra_flag=0,
    )
    score_crit, _ = calculate_population_priority_score(
        water_depth_cm=45.0,
        severity="Medium",
        population_exposure=1500000.0,
        critical_infra_flag=1,
    )
    assert score_crit == score_normal + 20.0


def test_9_alerts_endpoint_valid_response():
    res = client.get(f"/alerts?scenario=HEAVY&timestep=T+2&bbox={DELHI_BBOX_STR}")
    assert res.status_code == 200
    data = res.json()
    assert data["basis"] == "model_derived_forecast_triage"
    assert data["scenario"] == "HEAVY"
    assert data["timestep"] == "T+2"
    assert "incidents" in data
    assert isinstance(data["incidents"], list)


def test_10_alerts_count_capped():
    res = client.get(f"/alerts?scenario=HEAVY&timestep=T+2&bbox={DELHI_BBOX_STR}")
    assert res.status_code == 200
    data = res.json()
    assert len(data["incidents"]) <= 50
    assert data["max_incidents_cap"] == 50


def test_11_alerts_required_fields():
    res = client.get(f"/alerts?scenario=HEAVY&timestep=T+2&bbox={DELHI_BBOX_STR}")
    assert res.status_code == 200
    data = res.json()
    assert len(data["incidents"]) > 0

    first = data["incidents"][0]
    required_keys = {
        "incident_id",
        "road_id",
        "severity",
        "water_depth_cm",
        "priority_level",
        "population_exposure",
        "critical_infra_flag",
        "forecast_timestep",
        "scenario",
        "recommended_action",
        "coordinates",
        "basis",
    }
    for key in required_keys:
        assert key in first


def test_12_alert_severity_consistency():
    res = client.get(f"/alerts?scenario=HEAVY&timestep=T+2&bbox={DELHI_BBOX_STR}")
    assert res.status_code == 200
    data = res.json()
    for inc in data["incidents"]:
        assert inc["severity"] in ("Low", "Medium", "High")
        assert inc["basis"] == "model_derived_forecast_triage"


def test_13_alert_timestep_matches():
    res = client.get(f"/alerts?scenario=HEAVY&timestep=T+3&bbox={DELHI_BBOX_STR}")
    assert res.status_code == 200
    data = res.json()
    assert data["timestep"] == "T+3"
    for inc in data["incidents"]:
        assert inc["forecast_timestep"] == "T+3"
