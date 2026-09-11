"""
AquaG FastAPI Backend API
Stage 7: Complete Backend Integration for AquaG / AquaGraph (SIH26085).

Endpoints:
- GET /health: Component health status (model, router, DEM, drainage).
- POST /predict: XGBoost Model V2 flood severity prediction & deterministic Action Priority.
- POST /flood_info: GIS point query (DEM elevation & MPD-1976 drainage location proximity).
- POST /route: Authoritative AquaGraph A* risk-aware road routing.
- GET /zones: Sample zone severity predictions for GIS display.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Literal, Dict, Any, List

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Import deterministic Action Priority Engine
import importlib.util, sys
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
action_path = project_root / "backend" / "action_priority.py"
spec = importlib.util.spec_from_file_location("action_priority", str(action_path))
action_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(action_mod)
calculate_action_priority = action_mod.calculate_action_priority

# Import backend modules
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from flood_map import get_flood_info, DEMDataset, DRAINS_GDF
from routing import AquaGRouter
from schemas import (
    FeatureInput,
    ActionPriority,
    PredictResponse,
    ZoneResponse,
    FloodInfoRequest,
    FloodInfoResponse,
    RouteRequest,
    RouteResponse,
    HealthResponse,
)

# Instantiate authoritative AquaGRouter ONCE at application startup
ROUTER = AquaGRouter()

# ---------------------------------------------------------------------------
# Paths & Model Bundle
# ---------------------------------------------------------------------------
MODEL_PATH = project_root / "models" / "aquag_model_v2.pkl"
METADATA_PATH = project_root / "models" / "aquag_model_v2_metadata.json"
ZONE_DATASET_PATH = project_root / "data" / "processed" / "aquag_ml_dataset_v2_renamed.csv"


def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")
    model = joblib.load(MODEL_PATH)
    import json
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return {"model": model, "metadata": meta}


BUNDLE = load_model_bundle()
MODEL = BUNDLE["model"]
METADATA = BUNDLE["metadata"]
FEATURE_ORDER = METADATA["feature_order"]
MODEL_TYPE = METADATA.get("model_type", "unknown")
MODEL_VERSION = METADATA.get("model_version", "unknown")

if ZONE_DATASET_PATH.exists():
    ZONE_DATA = pd.read_csv(ZONE_DATASET_PATH)
else:
    ZONE_DATA = pd.DataFrame()

# ---------------------------------------------------------------------------
# FastAPI App & CORS Configuration
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AquaG Flood Prediction & Risk Routing API",
    description="""
    **AquaG / AquaGraph (SIH26085) Urban Flood Nowcasting System Backend**

    Architecture:
    - **Model V2 (XGBoost)**: Zone-level flood severity classification (`Low`, `Medium`, `High`) based on 11 features.
    - **Action Priority Engine**: Deterministic operational action scoring (`IMMEDIATE ACTION`, `HIGH PRIORITY`, `MONITOR`, `NORMAL`).
    - **GIS Flood Info**: DEM ground elevation & MPD-1976 drainage location proximity.
    - **AquaGraph A***: Spatially varying risk-aware road routing with cKDTree nearest-node snapping.
    """,
    version="2.0.0",
)

# Configurable CORS origins via environment variable
raw_origin_env = os.getenv("AQUAG_FRONTEND_ORIGIN", "")
if raw_origin_env.strip():
    allowed_origins = [o.strip() for o in raw_origin_env.split(",") if o.strip()]
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def build_feature_df(payload: FeatureInput) -> pd.DataFrame:
    data = payload.model_dump()
    ordered = [data[col] for col in FEATURE_ORDER]
    return pd.DataFrame([ordered], columns=FEATURE_ORDER)


def predict_severity(payload: FeatureInput) -> PredictResponse:
    df = build_feature_df(payload)
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(df)[0]
        class_names = METADATA.get("class_names", ["Low", "Medium", "High"])
        prob_dict = {name: float(p) for name, p in zip(class_names, probs)}
        pred_idx = int(np.argmax(probs))
        severity = class_names[pred_idx]
    else:
        pred_idx = int(MODEL.predict(df)[0])
        class_names = METADATA.get("class_names", ["Low", "Medium", "High"])
        severity = class_names[pred_idx]
        prob_dict = None

    action = calculate_action_priority(
        flood_severity=severity,
        population_total=payload.population_total,
        critical_infra_flag=payload.critical_infra_flag,
        rainfall_1h=payload.rainfall_1h,
        distance_to_drain=payload.distance_to_drain,
    )
    return PredictResponse(
        flood_severity=severity,
        model_version=MODEL_VERSION,
        probabilities=prob_dict,
        action_priority=ActionPriority(**action),
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Backend Health Check",
    tags=["System"],
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        model_type=MODEL_TYPE,
        router_loaded=bool(ROUTER is not None and len(ROUTER.nodes) > 0),
        dem_loaded=bool(DEMDataset is not None),
        drainage_loaded=bool(DRAINS_GDF is not None and not DRAINS_GDF.empty),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Zone Flood Severity & Action Priority",
    tags=["Machine Learning"],
)
def predict(payload: FeatureInput) -> PredictResponse:
    return predict_severity(payload)


@app.get(
    "/zones",
    response_model=List[ZoneResponse],
    summary="Get Zone Flood Severity Samples",
    tags=["GIS & Zones"],
)
def zones() -> List[ZoneResponse]:
    if ZONE_DATA.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Zone dataset not found"
        )
    samples = ZONE_DATA.head(8).reset_index(drop=True)
    results: List[ZoneResponse] = []
    dummy_coords = [(0.0, 0.0)] * len(samples)
    for idx, row in samples.iterrows():
        payload_dict = {col: row[col] for col in FEATURE_ORDER if col in row}
        payload = FeatureInput(**payload_dict)  # type: ignore[arg-type]
        severity = predict_severity(payload).flood_severity
        results.append(
            ZoneResponse(
                zone_id=str(row.get("zone_id", idx)),
                latitude=dummy_coords[idx][0],
                longitude=dummy_coords[idx][1],
                severity=severity,
            )
        )
    return results


@app.post(
    "/flood_info",
    response_model=FloodInfoResponse,
    summary="Get Point Elevation & Drainage Proximity",
    tags=["GIS & Zones"],
)
def flood_info(payload: FloodInfoRequest) -> FloodInfoResponse:
    raw_res = get_flood_info(payload.lat, payload.lon)
    return FloodInfoResponse(
        latitude=raw_res.get("latitude", payload.lat),
        longitude=raw_res.get("longitude", payload.lon),
        elevation=raw_res.get("elevation"),
        elevation_m=raw_res.get("elevation_m"),
        nearest_drain=raw_res.get("nearest_drain"),
        nearest_drain_distance_m=raw_res.get("nearest_drain_distance_m"),
        risk_basis="spatial_proxy",
    )


@app.post(
    "/route",
    response_model=RouteResponse,
    summary="Calculate Risk-Aware Road Route",
    tags=["Routing Engine"],
)
def route(payload: RouteRequest) -> RouteResponse:
    res = ROUTER.route(
        payload.start_lat,
        payload.start_lon,
        payload.end_lat,
        payload.end_lon,
        risk=payload.risk,
    )

    if res.get("status") == "error":
        return RouteResponse(
            status="error",
            message=res.get("message", "Routing failed"),
            risk_mode=payload.risk,
            risk_level=payload.risk,
            risk_basis="spatial_proxy",
        )

    distance_m = res.get("distance_m", 0.0)
    return RouteResponse(
        status="ok",
        message=None,
        routing_mode=res.get("routing_mode", "flood_aware"),
        algorithm=res.get("algorithm", "A*"),
        risk_mode=payload.risk,
        risk_level=payload.risk,
        risk_multiplier=res.get("risk_multiplier", 1.0),
        risk_basis="spatial_proxy",
        start_node=res.get("start_node"),
        end_node=res.get("end_node"),
        origin_snap_distance_m=res.get("origin_snap_distance_m"),
        destination_snap_distance_m=res.get("destination_snap_distance_m"),
        distance_m=distance_m,
        physical_distance_m=distance_m,
        estimated_cost=res.get("estimated_cost"),
        routing_cost=res.get("routing_cost"),
        nodes_in_route=res.get("nodes_in_route"),
        route=res.get("route"),
        coordinates=res.get("coordinates"),
    )