"""
AquaG Alerts & Triage Engine (Stage 12 Phase 3B)

Generates model-derived forecast incident triage reports from street-level waterlogging
and population priority outputs.

IMPORTANT SCIENTIFIC DISCLAIMER:
Incidents represent MODEL-DERIVED FORECAST INCIDENTS derived from scenario simulation logic.
They DO NOT represent live confirmed emergency dispatch events.
"""

from __future__ import annotations
import time
from typing import Dict, Any, List, Optional
from population_priority import get_population_priority_geojson


def generate_recommended_action(
    priority_level: str,
    critical_infra_flag: int,
    water_depth_cm: float,
) -> str:
    """Deterministic recommended operational action text."""
    if priority_level == "CRITICAL" or water_depth_cm > 100.0:
        base_action = "Immediate field inspection and traffic intervention required."
    elif priority_level == "HIGH" or water_depth_cm > 50.0:
        base_action = "Dispatch field team and monitor drainage conditions."
    elif priority_level == "MEDIUM" or water_depth_cm > 25.0:
        base_action = "Monitor location and prepare response resources."
    else:
        base_action = "Continue monitoring."

    if critical_infra_flag == 1:
        base_action += " Prioritize inspection because critical infrastructure is exposed."

    return base_action


def get_alerts_triage(
    scenario: str = "NORMAL",
    timestep: str = "T+0",
    rainfall_1h: Optional[float] = None,
    rainfall_3h: Optional[float] = None,
    rainfall_6h: Optional[float] = None,
    recent_rainfall_intensity: Optional[float] = None,
    bbox: Optional[List[float]] = None,
    max_incidents: int = 50,
) -> Dict[str, Any]:
    """
    Generate model-derived forecast incident triage list.
    """
    t_start = time.time()
    timestep = str(timestep).upper().replace(" ", "+").strip()
    scenario = str(scenario).upper().strip()

    # Get population priority features
    p_geojson = get_population_priority_geojson(
        scenario=scenario,
        timestep=timestep,
        rainfall_1h=rainfall_1h,
        rainfall_3h=rainfall_3h,
        rainfall_6h=rainfall_6h,
        recent_rainfall_intensity=recent_rainfall_intensity,
        bbox=bbox,
        max_features=2500,
    )

    features = p_geojson.get("features", [])
    candidate_incidents = []

    for idx, f in enumerate(features):
        props = f["properties"]
        water_depth_cm = float(props.get("water_depth_cm", 0.0))
        severity = str(props.get("severity", "Low"))
        crit_flag = int(props.get("critical_infra_flag", 0))
        pop_exp = int(props.get("population_exposure", 0))
        priority_level = str(props.get("priority_level", "LOW"))
        priority_score = float(props.get("priority_score", 0.0))

        # Filter meaningful incident conditions
        is_high_water = water_depth_cm >= 25.0 or severity in ("High", "Medium")
        is_crit_infra_exposed = crit_flag == 1 and water_depth_cm >= 10.0
        is_high_pop_exposed = pop_exp >= 2000000 and water_depth_cm >= 15.0

        if not (is_high_water or is_crit_infra_exposed or is_high_pop_exposed):
            continue

        rec_action = generate_recommended_action(
            priority_level=priority_level,
            critical_infra_flag=crit_flag,
            water_depth_cm=water_depth_cm,
        )

        candidate_incidents.append({
            "incident_id": f"INC-{1001 + idx}",
            "road_id": props.get("road_id", f"road_{idx}"),
            "severity": severity,
            "water_depth_cm": water_depth_cm,
            "priority_level": priority_level,
            "priority_score": priority_score,
            "population_exposure": pop_exp,
            "critical_infra_flag": crit_flag,
            "forecast_timestep": timestep,
            "scenario": scenario,
            "district": props.get("district", "Delhi District"),
            "recommended_action": rec_action,
            "coordinates": f["geometry"]["coordinates"],
            "basis": "model_derived_forecast_triage",
        })

    # Sort incidents: CRITICAL -> HIGH -> MEDIUM -> LOW, then water depth desc, pop exp desc
    level_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    candidate_incidents.sort(
        key=lambda x: (
            level_order.get(x["priority_level"], 4),
            -x["water_depth_cm"],
            -x["population_exposure"],
            -x["critical_infra_flag"],
        )
    )

    final_incidents = candidate_incidents[:max_incidents]
    exec_ms = round((time.time() - t_start) * 1000.0, 1)

    return {
        "scenario": scenario,
        "timestep": timestep,
        "basis": "model_derived_forecast_triage",
        "total_incidents": len(candidate_incidents),
        "returned_incidents": len(final_incidents),
        "max_incidents_cap": max_incidents,
        "execution_ms": exec_ms,
        "incidents": final_incidents,
    }
