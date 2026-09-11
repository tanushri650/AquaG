import json
import pathlib
import numpy as np
import pandas as pd
from typing import Literal, Dict

# Load processed V2 dataset to compute normalization ranges (once at import)
DATASET_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed" / "aquag_ml_dataset_v2_renamed.csv"
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found for Action Priority Engine: {DATASET_PATH}")
_df = pd.read_csv(DATASET_PATH)

# Compute min/max for normalization
POP_MIN = _df["population_total"].min()
POP_MAX = _df["population_total"].max()
RAINFALL1_MIN = _df["rainfall_1h"].min()
RAINFALL1_MAX = _df["rainfall_1h"].max()
DIST_DRN_MIN = _df["distance_to_drain"].min()
DIST_DRN_MAX = _df["distance_to_drain"].max()

def _normalize(value: float, v_min: float, v_max: float, scale: float) -> float:
    """Linear normalization to 0‑scale, clipping to [0, scale]."""
    if v_max == v_min:
        return 0.0
    norm = (value - v_min) / (v_max - v_min) * scale
    return max(0.0, min(scale, norm))

def _inverse_normalize(value: float, v_min: float, v_max: float, scale: float) -> float:
    """Inverse linear normalization for distance (closer → higher score)."""
    if v_max == v_min:
        return 0.0
    norm = (v_max - value) / (v_max - v_min) * scale
    return max(0.0, min(scale, norm))

def calculate_action_priority(
    flood_severity: Literal["Low", "Medium", "High"],
    population_total: float,
    critical_infra_flag: int,
    rainfall_1h: float,
    distance_to_drain: float,
) -> Dict[str, object]:
    """Deterministic Action Priority Engine.

    Returns a dictionary with ``priority_score`` (0‑100 int), ``action_level``
    and ``recommended_action``.
    The algorithm is fully deterministic and based only on verified input
    features – no pump information is used.
    """
    # Validate inputs
    if flood_severity not in {"Low", "Medium", "High"}:
        raise ValueError("Invalid flood_severity value")
    if critical_infra_flag not in (0, 1):
        raise ValueError("critical_infra_flag must be 0 or 1")
    for name, val in {
        "population_total": population_total,
        "rainfall_1h": rainfall_1h,
        "distance_to_drain": distance_to_drain,
    }.items():
        if not isinstance(val, (int, float, np.number)) or not np.isfinite(val):
            raise ValueError(f"{name} must be a finite number")

    # Base score
    score = 0
    # Flood severity contribution
    severity_map = {"Low": 10, "Medium": 30, "High": 50}
    score += severity_map[flood_severity]
    # Critical infrastructure contribution
    if critical_infra_flag == 1:
        score += 25
    # Population contribution (0‑20 points)
    score += _normalize(population_total, POP_MIN, POP_MAX, 20)
    # Rainfall 1h contribution (0‑15 points)
    score += _normalize(rainfall_1h, RAINFALL1_MIN, RAINFALL1_MAX, 15)
    # Distance to drain contribution (closer = higher, 0‑10 points)
    score += _inverse_normalize(distance_to_drain, DIST_DRN_MIN, DIST_DRN_MAX, 10)
    # Clamp final score
    priority_score = int(round(max(0, min(100, score))))

    # Determine action level
    if priority_score >= 75:
        action_level = "IMMEDIATE ACTION"
        recommended_action = "Prioritize flood response and inspect nearby drainage/infrastructure."
    elif priority_score >= 50:
        action_level = "HIGH PRIORITY"
        recommended_action = "Deploy monitoring and prepare drainage response."
    elif priority_score >= 25:
        action_level = "MONITOR"
        recommended_action = "Continue monitoring rainfall and flood conditions."
    else:
        action_level = "NORMAL"
        recommended_action = "No immediate intervention required."

    return {
        "priority_score": priority_score,
        "action_level": action_level,
        "recommended_action": recommended_action,
    }
