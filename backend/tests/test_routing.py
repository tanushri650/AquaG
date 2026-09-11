from fastapi.testclient import TestClient
import sys
from pathlib import Path
import pytest

# Add backend directory to sys.path so we can import main
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from main import app, ROUTER
from routing import AquaGRouter

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "AquaG Model V2"
    assert data["model_type"] == "XGBoost"
    assert data["router_loaded"] is True
    assert data["dem_loaded"] is True
    assert data["drainage_loaded"] is True


def test_predict_valid_request():
    payload = {
        "rainfall_1h": 10.0,
        "rainfall_3h": 20.0,
        "rainfall_6h": 30.0,
        "recent_rainfall_intensity": 15.0,
        "elevation": 210.0,
        "slope": 1.5,
        "distance_to_drain": 100.0,
        "distance_to_road": 50.0,
        "distance_to_infra": 500.0,
        "population_total": 100000.0,
        "critical_infra_flag": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["flood_severity"] in ["Low", "Medium", "High"]
    assert data["model_version"] == "AquaG Model V2"
    assert "probabilities" in data
    assert "action_priority" in data
    action = data["action_priority"]
    assert action["action_level"] in [
        "IMMEDIATE ACTION",
        "HIGH PRIORITY",
        "MONITOR",
        "NORMAL",
    ]
    assert isinstance(action["priority_score"], int)


def test_predict_invalid_request_missing_features():
    payload = {"rainfall_1h": 10.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_request_critical_infra_flag():
    payload = {
        "rainfall_1h": 10.0,
        "rainfall_3h": 20.0,
        "rainfall_6h": 30.0,
        "recent_rainfall_intensity": 15.0,
        "elevation": 210.0,
        "slope": 1.5,
        "distance_to_drain": 100.0,
        "distance_to_road": 50.0,
        "distance_to_infra": 500.0,
        "population_total": 100000.0,
        "critical_infra_flag": 5,  # Must be 0 or 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_flood_info_valid_request():
    response = client.post("/flood_info", json={"lat": 28.6139, "lon": 77.2090})
    assert response.status_code == 200
    data = response.json()
    assert data["latitude"] == 28.6139
    assert data["longitude"] == 77.2090
    assert data["risk_basis"] == "spatial_proxy"
    assert "elevation" in data
    assert "nearest_drain" in data


def test_flood_info_invalid_coordinates():
    response = client.post("/flood_info", json={"lat": 999.0, "lon": 77.2090})
    assert response.status_code == 422


def test_route_valid_request():
    payload = {
        "start_lat": 28.6139,
        "start_lon": 77.2090,
        "end_lat": 28.6500,
        "end_lon": 77.2300,
        "risk": "Low",
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["routing_mode"] == "flood_aware"
    assert data["algorithm"] == "A*"
    assert data["risk_basis"] == "spatial_proxy"
    assert data["distance_m"] > 0
    assert data["physical_distance_m"] == data["distance_m"]
    assert len(data["route"]) > 1
    assert len(data["coordinates"]) > 1


def test_route_invalid_risk_mode():
    payload = {
        "start_lat": 28.6139,
        "start_lon": 77.2090,
        "end_lat": 28.6500,
        "end_lon": 77.2300,
        "risk": "Extreme",  # Invalid
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 422


def test_route_invalid_coordinates():
    payload = {
        "start_lat": 999.0,
        "start_lon": 77.2090,
        "end_lat": 28.6500,
        "end_lon": 77.2300,
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 422


def test_route_outside_domain():
    payload = {
        "start_lat": 40.7128,
        "start_lon": -74.0060,
        "end_lat": 40.7306,
        "end_lon": -73.9352,
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert "domain" in data["message"].lower()


def test_router_initialization_and_cKDTree():
    assert isinstance(ROUTER, AquaGRouter)
    assert ROUTER.node_count == 651271
    node_id, snap_dist = ROUTER.nearest_node(28.6139, 77.2090)
    assert node_id in (12434944735, 626651)
    assert round(snap_dist, 2) in (2.08, 1.93)





def test_end_to_end_pipeline():
    # 1. Predict
    pred_res = client.post(
        "/predict",
        json={
            "rainfall_1h": 50.0,
            "rainfall_3h": 80.0,
            "rainfall_6h": 120.0,
            "recent_rainfall_intensity": 25.0,
            "elevation": 208.0,
            "slope": 2.0,
            "distance_to_drain": 40.0,
            "distance_to_road": 30.0,
            "distance_to_infra": 100.0,
            "population_total": 150000.0,
            "critical_infra_flag": 1,
        },
    ).json()
    assert pred_res["flood_severity"] in ["Low", "Medium", "High"]
    assert pred_res["action_priority"]["action_level"] in [
        "IMMEDIATE ACTION",
        "HIGH PRIORITY",
        "MONITOR",
        "NORMAL",
    ]

    # 2. Flood Info
    info_res = client.post(
        "/flood_info", json={"lat": 28.6139, "lon": 77.2090}
    ).json()
    assert info_res["risk_basis"] == "spatial_proxy"

    # 3. Route
    route_res = client.post(
        "/route",
        json={
            "start_lat": 28.6139,
            "start_lon": 77.2090,
            "end_lat": 28.6500,
            "end_lon": 77.2300,
            "risk": "High",
        },
    ).json()
    assert route_res["status"] == "ok"
    assert route_res["risk_basis"] == "spatial_proxy"
