"""
Phase 3C Backend Unit Tests — Dynamic Flood-Aware AquaGraph Routing

Verifies dynamic route calculations, scenario/timestep responsiveness,
avoidance of heavily flooded road segments, scientific labeling, and backward compatibility.
"""

import sys
import pathlib
import pytest
from fastapi.testclient import TestClient

# Path resolution for imports
backend_dir = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir))

from main import app
from routing import AquaGRouter, BASE_RISK_MULTIPLIERS

client = TestClient(app)

# Coordinates in Delhi domain
START_LAT, START_LON = 28.613900, 77.209000
END_LAT, END_LON = 28.650000, 77.230000


def test_1_existing_route_request_backward_compatible():
    """Legacy route request format without scenario/timestep parameters must pass."""
    payload = {
        "start_lat": START_LAT,
        "start_lon": START_LON,
        "end_lat": END_LAT,
        "end_lon": END_LON,
        "risk": "Low",
    }
    res = client.post("/route", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["distance_m"] > 0
    assert "routing_cost" in data
    assert "coordinates" in data


def test_2_flood_aware_route_request():
    """Dynamic flood-aware route request must return valid Phase 3C metadata."""
    payload = {
        "start_lat": START_LAT,
        "start_lon": START_LON,
        "end_lat": END_LAT,
        "end_lon": END_LON,
        "risk": "Low",
        "scenario": "HEAVY",
        "timestep": "T+2",
        "flood_aware": True,
    }
    res = client.post("/route", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["flood_aware"] is True
    assert data["scenario"] == "HEAVY"
    assert data["timestep"] == "T+2"
    assert data["basis"] == "model_derived_flood_aware_routing"


def test_3_scenario_and_timestep_parameters_accepted():
    """Scenario and timestep strings must be passed and normalized in response."""
    payload = {
        "start_lat": START_LAT,
        "start_lon": START_LON,
        "end_lat": END_LAT,
        "end_lon": END_LON,
        "scenario": "extreme",
        "timestep": "T+3",
    }
    res = client.post("/route", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["scenario"] == "EXTREME"
    assert data["timestep"] == "T+3"


def test_4_dynamic_routing_cost_and_depth_metadata():
    """Route response must contain maximum water depth, flooded segment count, and avoided segments."""
    payload = {
        "start_lat": START_LAT,
        "start_lon": START_LON,
        "end_lat": END_LAT,
        "end_lon": END_LON,
        "scenario": "HEAVY",
        "timestep": "T+2",
        "flood_aware": True,
    }
    res = client.post("/route", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "maximum_water_depth_cm" in data
    assert "flooded_segments_on_route" in data
    assert "route_risk_level" in data
    assert "avoided_high_risk_segments" in data

    assert isinstance(data["maximum_water_depth_cm"], (int, float))
    assert isinstance(data["flooded_segments_on_route"], int)
    assert data["route_risk_level"] in ("Low", "Medium", "High", "Critical")
    assert isinstance(data["avoided_high_risk_segments"], int)


def test_5_high_water_increases_routing_cost():
    """Routing cost under HEAVY scenario must be greater than or equal to NORMAL scenario."""
    res_normal = client.post("/route", json={
        "start_lat": START_LAT,
        "start_lon": START_LON,
        "end_lat": END_LAT,
        "end_lon": END_LON,
        "scenario": "NORMAL",
        "timestep": "T+0",
    })
    res_heavy = client.post("/route", json={
        "start_lat": START_LAT,
        "start_lon": START_LON,
        "end_lat": END_LAT,
        "end_lon": END_LON,
        "scenario": "EXTREME",
        "timestep": "T+3",
    })

    assert res_normal.status_code == 200
    assert res_heavy.status_code == 200

    cost_normal = res_normal.json()["routing_cost"]
    cost_heavy = res_heavy.json()["routing_cost"]

    assert cost_heavy >= cost_normal


def test_6_over_100cm_extreme_penalty():
    """Segments > 100 cm depth must yield an extreme routing penalty multiplier (>= 100.0)."""
    router = AquaGRouter()
    penalty_shallow = router.calculate_flood_penalty_multiplier(5.0)
    penalty_mod = router.calculate_flood_penalty_multiplier(20.0)
    penalty_deep = router.calculate_flood_penalty_multiplier(60.0)
    penalty_extreme = router.calculate_flood_penalty_multiplier(110.0)

    assert penalty_shallow == 1.0
    assert 1.5 <= penalty_mod <= 2.5
    assert 8.0 <= penalty_deep <= 20.0
    assert penalty_extreme >= 100.0


def test_7_longer_safer_route_selection():
    """
    Deterministic test: Verifies that flood penalty multiplier makes a 1500m clear route (cost ~1500)
    vastly superior to a 1000m heavily flooded route (80cm depth, cost ~15200).
    """
    router = AquaGRouter()
    distance_short_flooded = 1000.0
    depth_short_flooded = 80.0
    mult_short = router.calculate_flood_penalty_multiplier(depth_short_flooded)
    cost_short = distance_short_flooded * mult_short

    distance_long_clear = 1500.0
    depth_long_clear = 2.0
    mult_long = router.calculate_flood_penalty_multiplier(depth_long_clear)
    cost_long = distance_long_clear * mult_long

    # The cost of the longer safer route must be strictly less than the shorter flooded route
    assert cost_long < cost_short
    assert mult_short > 10.0
    assert mult_long == 1.0


def test_8_regression_all_previous_endpoints():
    """Regression check: Ensures all previous phase endpoints remain 100% operational."""
    # 1. Health
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    # 2. Predict
    r_pred = client.post("/predict", json={
        "rainfall_1h": 45.0, "rainfall_3h": 85.0, "rainfall_6h": 130.0,
        "recent_rainfall_intensity": 22.5, "elevation": 208.5, "slope": 1.8,
        "distance_to_drain": 45.0, "distance_to_road": 20.0, "distance_to_infra": 150.0,
        "population_total": 120000, "critical_infra_flag": 1
    })
    assert r_pred.status_code == 200

    # 3. Waterlogging
    r_wl = client.post("/waterlogging", json={
        "scenario": "HEAVY", "timestep": "T+2", "rainfall_1h": 75.0,
        "rainfall_3h": 130.0, "rainfall_6h": 180.0, "recent_rainfall_intensity": 37.5,
        "bbox": [77.15, 28.55, 77.25, 28.65]
    })
    assert r_wl.status_code == 200

    # 4. Infrastructure
    r_inf = client.get("/infrastructure?bbox=77.15,28.55,77.25,28.65")
    assert r_inf.status_code == 200

    # 5. Drainage
    r_drn = client.get("/drainage?bbox=77.15,28.55,77.25,28.65")
    assert r_drn.status_code == 200

    # 6. Pumps
    r_pmp = client.get("/pumps")
    assert r_pmp.status_code == 200

    # 7. Population Priority
    r_pop = client.get("/population-priority?scenario=HEAVY&timestep=T+2&bbox=77.15,28.55,77.25,28.65")
    assert r_pop.status_code == 200

    # 8. Alerts
    r_alt = client.get("/alerts?scenario=HEAVY&timestep=T+2&bbox=77.15,28.55,77.25,28.65")
    assert r_alt.status_code == 200
