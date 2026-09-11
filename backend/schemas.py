"""
Pydantic Schemas for AquaG FastAPI Backend
Centralized request and response validation models.
"""

from __future__ import annotations
from typing import Literal, Dict, Any, List
import numpy as np
from pydantic import BaseModel, Field, field_validator


class FeatureInput(BaseModel):
    rainfall_1h: float
    rainfall_3h: float
    rainfall_6h: float
    recent_rainfall_intensity: float
    elevation: float
    slope: float
    distance_to_drain: float
    distance_to_road: float
    distance_to_infra: float
    population_total: float
    critical_infra_flag: int = Field(..., ge=0, le=1)

    @field_validator(
        "rainfall_1h",
        "rainfall_3h",
        "rainfall_6h",
        "recent_rainfall_intensity",
        "elevation",
        "slope",
        "distance_to_drain",
        "distance_to_road",
        "distance_to_infra",
        "population_total",
        mode="before",
    )

    @classmethod
    def ensure_finite(cls, v: Any) -> Any:
        if isinstance(v, (int, float)) and not np.isfinite(v):
            raise ValueError("Numeric values must be finite")
        return v


class ActionPriority(BaseModel):
    priority_score: int
    action_level: Literal["IMMEDIATE ACTION", "HIGH PRIORITY", "MONITOR", "NORMAL"]
    recommended_action: str


class PredictResponse(BaseModel):
    flood_severity: Literal["Low", "Medium", "High"]
    model_version: str
    probabilities: Dict[str, float] | None = None
    action_priority: ActionPriority


class ZoneResponse(BaseModel):
    zone_id: str
    latitude: float
    longitude: float
    severity: Literal["Low", "Medium", "High"]


class FloodInfoRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in degrees")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude in degrees")

    @field_validator("lat", "lon", mode="before")

    @classmethod
    def ensure_finite_coord(cls, v: Any) -> Any:
        if isinstance(v, (int, float)) and not np.isfinite(v):
            raise ValueError("Coordinate values must be finite")
        return v


class FloodInfoResponse(BaseModel):
    latitude: float | None = None
    longitude: float | None = None
    elevation: float | None = None
    elevation_m: float | None = None
    nearest_drain: dict | None = None
    nearest_drain_distance_m: float | None = None
    risk_basis: str = "spatial_proxy"


class RouteRequest(BaseModel):
    start_lat: float = Field(..., ge=-90.0, le=90.0)
    start_lon: float = Field(..., ge=-180.0, le=180.0)
    end_lat: float = Field(..., ge=-90.0, le=90.0)
    end_lon: float = Field(..., ge=-180.0, le=180.0)
    risk: Literal["Low", "Medium", "High"] = "Low"
    scenario: str = "NORMAL"
    timestep: str = "T+0"
    rainfall_1h: float | None = None
    rainfall_3h: float | None = None
    rainfall_6h: float | None = None
    recent_rainfall_intensity: float | None = None
    flood_aware: bool = True

    @field_validator("start_lat", "start_lon", "end_lat", "end_lon", mode="before")
    @classmethod
    def ensure_finite_route_coords(cls, v: Any) -> Any:
        if isinstance(v, (int, float)) and not np.isfinite(v):
            raise ValueError("Route coordinates must be finite")
        return v


class RouteResponse(BaseModel):
    status: str
    message: str | None = None
    routing_mode: str | None = None
    algorithm: str | None = None
    risk_mode: str | None = None
    risk_level: str | None = None
    risk_multiplier: float | None = None
    risk_basis: str | None = None
    start_node: int | None = None
    end_node: int | None = None
    origin_snap_distance_m: float | None = None
    destination_snap_distance_m: float | None = None
    distance_m: float | None = None
    physical_distance_m: float | None = None
    estimated_cost: float | None = None
    routing_cost: float | None = None
    nodes_in_route: int | None = None
    route: list[dict] | None = None
    coordinates: list[list[float]] | None = None
    flood_aware: bool | None = None
    scenario: str | None = None
    timestep: str | None = None
    route_risk_level: str | None = None
    flooded_segments_on_route: int | None = None
    maximum_water_depth_cm: float | None = None
    avoided_high_risk_segments: int | None = None
    basis: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_type: str
    router_loaded: bool
    dem_loaded: bool
    drainage_loaded: bool


class WaterloggingRequest(BaseModel):
    scenario: str = "MODERATE"
    timestep: str = "T+0"
    rainfall_1h: float | None = Field(None, ge=0.0)
    rainfall_3h: float | None = Field(None, ge=0.0)
    rainfall_6h: float | None = Field(None, ge=0.0)
    recent_rainfall_intensity: float | None = Field(None, ge=0.0)
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError("bbox must contain exactly 4 numbers: [min_lon, min_lat, max_lon, max_lat]")
        min_lon, min_lat, max_lon, max_lat = v
        if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
            raise ValueError("Bounding box longitudes must be between -180 and 180")
        if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
            raise ValueError("Bounding box latitudes must be between -90 and 90")
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("min_lon must be less than max_lon and min_lat must be less than max_lat")
        return v


class WaterloggingResponse(BaseModel):
    type: str = "FeatureCollection"
    features: List[Dict[str, Any]]
    metadata: Dict[str, Any]

