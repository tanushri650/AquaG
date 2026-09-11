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


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_type: str
    router_loaded: bool
    dem_loaded: bool
    drainage_loaded: bool
