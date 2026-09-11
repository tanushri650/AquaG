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
from waterlogging import get_street_waterlogging_geojson
from infrastructure import get_infrastructure_geojson
from pumps import get_pumps_metadata
from drainage import get_drainage_geojson
from population_priority import get_population_priority_geojson
from alerts import get_alerts_triage


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
    WaterloggingRequest,
    WaterloggingResponse,
)

# Instantiate authoritative AquaGRouter ONCE at application startup
ROUTER = AquaGRouter()

# ---------------------------------------------------------------------------
# Paths & Model Bundle
# ---------------------------------------------------------------------------
MODEL_PATH = project_root / "models" / "aquag_model_v2.pkl"
METADATA_PATH = project_root / "models" / "aquag_model_v2_metadata.json"
ZONE_DATASET_PATH = project_root / "data" / "processed" / "aquag_ml_dataset_v2_renamed.csv"


from shared_resources import get_shared_model_bundle

BUNDLE = get_shared_model_bundle()
MODEL = BUNDLE["model"]
METADATA = BUNDLE["metadata"]
FEATURE_ORDER = METADATA.get("feature_order", [])
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
        "https://tanushri650.github.io",
        "https://tanushri650.github.io/AquaG",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("aquag")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred.", "error": str(exc)},
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
    from flood_map import DEM_PATH, DRAINS_DIR
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        model_type=MODEL_TYPE,
        router_loaded=bool(ROUTER is not None and ROUTER.node_count > 0),
        dem_loaded=bool(DEM_PATH.exists()),
        drainage_loaded=bool(DRAINS_DIR.exists()),
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
        start_lat=payload.start_lat,
        start_lon=payload.start_lon,
        end_lat=payload.end_lat,
        end_lon=payload.end_lon,
        risk=payload.risk,
        scenario=payload.scenario,
        timestep=payload.timestep,
        rainfall_1h=payload.rainfall_1h,
        rainfall_3h=payload.rainfall_3h,
        rainfall_6h=payload.rainfall_6h,
        recent_rainfall_intensity=payload.recent_rainfall_intensity,
        flood_aware=payload.flood_aware,
    )

    if res.get("status") == "error":
        return RouteResponse(
            status="error",
            message=res.get("message", "Routing failed"),
            risk_mode=payload.risk,
            risk_level=payload.risk,
            risk_basis="spatial_proxy",
            basis="model_derived_flood_aware_routing" if payload.flood_aware else "spatial_proxy",
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
        flood_aware=res.get("flood_aware", payload.flood_aware),
        scenario=res.get("scenario", payload.scenario),
        timestep=res.get("timestep", payload.timestep),
        route_risk_level=res.get("route_risk_level", "Low"),
        flooded_segments_on_route=res.get("flooded_segments_on_route", 0),
        maximum_water_depth_cm=res.get("maximum_water_depth_cm", 0.0),
        avoided_high_risk_segments=res.get("avoided_high_risk_segments", 0),
        basis=res.get("basis", "model_derived_flood_aware_routing"),
    )


@app.post(
    "/waterlogging",
    response_model=WaterloggingResponse,
    summary="Get Viewport-Filtered Street-Level Waterlogging GeoJSON Layer",
    tags=["GIS & Waterlogging"],
)
def waterlogging(payload: WaterloggingRequest) -> WaterloggingResponse:
    res = get_street_waterlogging_geojson(
        scenario=payload.scenario,
        timestep=payload.timestep,
        rainfall_1h=payload.rainfall_1h,
        rainfall_3h=payload.rainfall_3h,
        rainfall_6h=payload.rainfall_6h,
        recent_rainfall_intensity=payload.recent_rainfall_intensity,
        bbox=payload.bbox,
    )
    return WaterloggingResponse(**res)


@app.get(
    "/infrastructure",
    summary="Get Viewport-Filtered Critical Infrastructure GeoJSON Layer",
    tags=["GIS & Infrastructure"],
)
def infrastructure(bbox: str | None = None) -> Dict[str, Any]:
    bbox_list = None
    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) == 4:
                bbox_list = parts
        except ValueError:
            pass
    return get_infrastructure_geojson(bbox=bbox_list)


@app.get(
    "/pumps",
    summary="Get Permanent Pumping Stations Source Metadata",
    tags=["GIS & Infrastructure"],
)
def pumps() -> Dict[str, Any]:
    return get_pumps_metadata()


@app.get(
    "/drainage",
    summary="Get Viewport-Filtered MPD-1976 Drainage Reference Points GeoJSON Layer",
    tags=["GIS & Infrastructure"],
)
def drainage(bbox: str | None = None) -> Dict[str, Any]:
    bbox_list = None
    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) == 4:
                bbox_list = parts
        except ValueError:
            pass
    return get_drainage_geojson(bbox=bbox_list)


@app.get(
    "/population-priority",
    summary="Get Viewport-Filtered Population Response Priority GeoJSON Layer",
    tags=["GIS & Population Priority"],
)
def population_priority(
    scenario: str = "NORMAL",
    timestep: str = "T+0",
    rainfall_1h: float | None = None,
    rainfall_3h: float | None = None,
    rainfall_6h: float | None = None,
    recent_rainfall_intensity: float | None = None,
    bbox: str | None = None,
) -> Dict[str, Any]:
    bbox_list = None
    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) == 4:
                bbox_list = parts
        except ValueError:
            pass
    return get_population_priority_geojson(
        scenario=scenario,
        timestep=timestep,
        rainfall_1h=rainfall_1h,
        rainfall_3h=rainfall_3h,
        rainfall_6h=rainfall_6h,
        recent_rainfall_intensity=recent_rainfall_intensity,
        bbox=bbox_list,
    )


@app.get(
    "/alerts",
    summary="Get Model-Derived Forecast Incident Triage List",
    tags=["Alerts & Triage"],
)
def alerts(
    scenario: str = "NORMAL",
    timestep: str = "T+0",
    rainfall_1h: float | None = None,
    rainfall_3h: float | None = None,
    rainfall_6h: float | None = None,
    recent_rainfall_intensity: float | None = None,
    bbox: str | None = None,
) -> Dict[str, Any]:
    bbox_list = None
    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) == 4:
                bbox_list = parts
        except ValueError:
            pass
    return get_alerts_triage(
        scenario=scenario,
        timestep=timestep,
        rainfall_1h=rainfall_1h,
        rainfall_3h=rainfall_3h,
        rainfall_6h=rainfall_6h,
        recent_rainfall_intensity=recent_rainfall_intensity,
        bbox=bbox_list,
    )