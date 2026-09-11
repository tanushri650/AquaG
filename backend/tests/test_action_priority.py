import math
import numpy as np
import pandas as pd
import pathlib

import importlib.util, pathlib, sys
backend_dir = pathlib.Path(__file__).resolve().parents[1]
module_path = backend_dir / "action_priority.py"
spec = importlib.util.spec_from_file_location("action_priority", str(module_path))
action_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(action_module)
calculate_action_priority = action_module.calculate_action_priority

# Helper to load dataset min/max for manual calculation (mirrors module logic)
DATASET_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed" / "aquag_ml_dataset_v2_renamed.csv"



_df = pd.read_csv(DATASET_PATH)
POP_MIN = _df["population_total"].min()
POP_MAX = _df["population_total"].max()
RAINFALL1_MIN = _df["rainfall_1h"].min()
RAINFALL1_MAX = _df["rainfall_1h"].max()
DIST_DRN_MIN = _df["distance_to_drain"].min()
DIST_DRN_MAX = _df["distance_to_drain"].max()

def _norm(val, vmin, vmax, scale):
    if vmax == vmin:
        return 0.0
    n = (val - vmin) / (vmax - vmin) * scale
    return max(0.0, min(scale, n))

def _inv_norm(val, vmin, vmax, scale):
    if vmax == vmin:
        return 0.0
    n = (vmax - val) / (vmax - vmin) * scale
    return max(0.0, min(scale, n))

def manual_score(sev, pop, crit, rain1h, dist):
    score = {"Low":10, "Medium":30, "High":50}[sev]
    if crit == 1:
        score += 25
    score += _norm(pop, POP_MIN, POP_MAX, 20)
    score += _norm(rain1h, RAINFALL1_MIN, RAINFALL1_MAX, 15)
    score += _inv_norm(dist, DIST_DRN_MIN, DIST_DRN_MAX, 10)
    return int(round(max(0, min(100, score))))

def test_high_critical_high_population():
    # Use max values for population and rainfall to ensure high score
    pop = POP_MAX
    rain = RAINFALL1_MAX
    dist = DIST_DRN_MIN  # closest to drain
    result = calculate_action_priority(
        flood_severity="High",
        population_total=pop,
        critical_infra_flag=1,
        rainfall_1h=rain,
        distance_to_drain=dist,
    )
    expected_score = manual_score("High", pop, 1, rain, dist)
    assert result["priority_score"] == expected_score
    assert result["action_level"] == "IMMEDIATE ACTION"
    assert "Prioritize flood response" in result["recommended_action"]

def test_medium_noncritical_low_population():
    pop = POP_MIN
    rain = RAINFALL1_MIN
    dist = DIST_DRN_MAX  # farthest
    result = calculate_action_priority(
        flood_severity="Medium",
        population_total=pop,
        critical_infra_flag=0,
        rainfall_1h=rain,
        distance_to_drain=dist,
    )
    expected_score = manual_score("Medium", pop, 0, rain, dist)
    assert result["priority_score"] == expected_score
    # Score should be low enough to be in MONITOR or NORMAL
    if expected_score >= 25:
        assert result["action_level"] == "MONITOR"
    else:
        assert result["action_level"] == "NORMAL"

def test_boundary_scores():
    # Construct inputs to hit exactly 75, 50, 25 thresholds using reverse engineering
    # We'll iterate over a small grid to find a combination that yields each boundary.
    def find_score(target):
        for pop in [POP_MIN, POP_MAX]:
            for rain in [RAINFALL1_MIN, RAINFALL1_MAX]:
                for dist in [DIST_DRN_MIN, DIST_DRN_MAX]:
                    for sev, base in [("High",50),("Medium",30),("Low",10)]:
                        for crit in [0,1]:
                            sc = manual_score(sev, pop, crit, rain, dist)
                            if sc == target:
                                return sev, pop, crit, rain, dist
        return None
    for target, level in [(75,"IMMEDIATE ACTION"),(50,"HIGH PRIORITY"),(25,"MONITOR")]:
        combo = find_score(target)
        assert combo is not None, f"Could not find combo for score {target}"
        sev,pop,crit,rain,dist = combo
        result = calculate_action_priority(
            flood_severity=sev,
            population_total=pop,
            critical_infra_flag=crit,
            rainfall_1h=rain,
            distance_to_drain=dist,
        )
        assert result["priority_score"] == target
        assert result["action_level"] == level

def test_invalid_severity():
    try:
        calculate_action_priority(
            flood_severity="Extreme",
            population_total=POP_MIN,
            critical_infra_flag=0,
            rainfall_1h=RAINFALL1_MIN,
            distance_to_drain=DIST_DRN_MAX,
        )
    except ValueError as e:
        assert "Invalid flood_severity" in str(e)
    else:
        assert False, "Expected ValueError for invalid severity"

def test_invalid_critical_flag():
    try:
        calculate_action_priority(
            flood_severity="Low",
            population_total=POP_MIN,
            critical_infra_flag=2,
            rainfall_1h=RAINFALL1_MIN,
            distance_to_drain=DIST_DRN_MAX,
        )
    except ValueError as e:
        assert "critical_infra_flag" in str(e)
    else:
        assert False, "Expected ValueError for invalid critical flag"

def test_nan_input():
    try:
        calculate_action_priority(
            flood_severity="Low",
            population_total=float('nan'),
            critical_infra_flag=0,
            rainfall_1h=RAINFALL1_MIN,
            distance_to_drain=DIST_DRN_MAX,
        )
    except ValueError as e:
        assert "population_total" in str(e)
    else:
        assert False, "Expected ValueError for NaN input"
