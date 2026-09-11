"""Train the AquaG flood-severity model from zone_flood_data.csv.

The input file is the validated 150-zone dataset supplied for this project.
The target is the source-provided flood_severity class (Low, Medium, High).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT.parent / "data" / "processed" / "zone_flood_data.csv"
MODEL_PATH = ROOT / "aquag_model.pkl"
VERSIONS_PATH = ROOT / "package_versions.txt"
REPORT_PATH = ROOT / "classification_report.txt"

EXPECTED_COLUMNS = [
    "zone_id",
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "elevation",
    "slope",
    "distance_to_drain",
    "drain_capacity",
    "pump_load_pct",
    "flood_history_count",
    "population_density",
    "critical_infra_flag",
    "flood_severity",
]
FEATURE_COLUMNS = EXPECTED_COLUMNS[1:-1]
CLASS_NAMES = ["High", "Low", "Medium"]
NUMERIC_COLUMNS = EXPECTED_COLUMNS[1:-1]


def validate_dataset(frame: pd.DataFrame) -> None:
    """Fail clearly when the supplied CSV does not match the requested spec."""

    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "zone_flood_data.csv columns do not match the required order: "
            f"{EXPECTED_COLUMNS}"
        )
    if len(frame) != 150:
        raise ValueError(f"Expected 150 rows, found {len(frame)}")
    if frame.isna().any().any():
        missing = frame.isna().sum()
        raise ValueError(f"Dataset contains missing values: {missing[missing > 0].to_dict()}")
    if not frame["zone_id"].is_unique:
        raise ValueError("zone_id values must be unique")
    if set(frame["flood_severity"]) != set(CLASS_NAMES):
        raise ValueError(
            "flood_severity must contain exactly Low, Medium, and High"
        )

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    if not (frame["rainfall_1h"] < frame["rainfall_3h"]).all():
        raise ValueError("rainfall_1h must be less than rainfall_3h in every row")
    if not (frame["rainfall_3h"] < frame["rainfall_6h"]).all():
        raise ValueError("rainfall_3h must be less than rainfall_6h in every row")
    if not frame["critical_infra_flag"].isin([0, 1]).all():
        raise ValueError("critical_infra_flag must contain only 0 or 1")
    if not frame["flood_history_count"].between(0, 5).all():
        raise ValueError("flood_history_count must be between 0 and 5")

    score = (
        frame["rainfall_3h"] * 0.4
        + frame["flood_history_count"] * 20
        - frame["elevation"] * 0.5
        - frame["drain_capacity"] * 0.3
    )
    calculated = score.map(
        lambda value: "High" if value > 60 else ("Medium" if value > 30 else "Low")
    )
    if not calculated.eq(frame["flood_severity"]).all():
        raise ValueError("flood_severity values do not match the required score formula")


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing required dataset: {DATASET_PATH}")
    frame = pd.read_csv(DATASET_PATH)
    validate_dataset(frame)
    return frame


def train_model(
    frame: pd.DataFrame,
) -> tuple[XGBClassifier, LabelEncoder, dict[str, object]]:
    encoder = LabelEncoder()
    encoded_target = encoder.fit_transform(frame["flood_severity"])
    features = frame[FEATURE_COLUMNS].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        encoded_target,
        test_size=0.2,
        random_state=42,
        stratify=encoded_target,
    )

    model = XGBClassifier(
        n_estimators=140,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(CLASS_NAMES),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test).astype(int)
    actual_names = encoder.inverse_transform(y_test)
    predicted_names = encoder.inverse_transform(predictions)
    report = classification_report(
        actual_names,
        predicted_names,
        labels=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(actual_names, predicted_names)),
        "classification_report": {
            name: {
                key: float(value)
                for key, value in values.items()
            }
            for name, values in report.items()
            if isinstance(values, dict)
        },
        "test_rows": int(len(y_test)),
    }
    return model, encoder, metrics


def save_outputs(
    frame: pd.DataFrame,
    model: XGBClassifier,
    encoder: LabelEncoder,
    metrics: dict[str, object],
) -> None:
    dataset_hash = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    joblib.dump(
        {
            "model": model,
            "label_encoder": encoder,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": "flood_severity",
            "classes": list(encoder.classes_),
            "dataset_sha256": dataset_hash,
            "training_rows": int(len(frame)),
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    report_lines = [
        f"Dataset SHA-256: {dataset_hash}",
        f"Training rows: {len(frame)}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        "",
        "Classification report:",
        json.dumps(metrics["classification_report"], indent=2),
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    VERSIONS_PATH.write_text(
        "\n".join(
            [
                f"python={__import__('sys').version.split()[0]}",
                f"pandas={pd.__version__}",
                f"numpy={np.__version__}",
                f"scikit-learn={sklearn.__version__}",
                f"xgboost={xgboost.__version__}",
                f"joblib={joblib.__version__}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_sanity_prediction(
    model: XGBClassifier,
    encoder: LabelEncoder,
) -> str:
    sample = pd.DataFrame(
        [
            {
                "rainfall_1h": 20,
                "rainfall_3h": 45,
                "rainfall_6h": 60,
                "elevation": 15,
                "slope": 2,
                "distance_to_drain": 100,
                "drain_capacity": 40,
                "pump_load_pct": 85,
                "flood_history_count": 3,
                "population_density": 12000,
                "critical_infra_flag": 1,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    encoded_prediction = model.predict(sample).astype(int)
    return str(encoder.inverse_transform(encoded_prediction)[0])


def main() -> None:
    frame = load_dataset()
    model, encoder, metrics = train_model(frame)
    save_outputs(frame, model, encoder, metrics)
    predicted_severity = run_sanity_prediction(model, encoder)

    print(f"Rows loaded: {len(frame)}")
    print(
        "Severity counts: "
        f"{frame['flood_severity'].value_counts().sort_index().to_dict()}"
    )
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Sanity prediction: {predicted_severity}")
    print(f"Saved model: {MODEL_PATH.name}")
    print(f"Saved report: {REPORT_PATH.name}")
    print(f"Saved versions: {VERSIONS_PATH.name}")


if __name__ == "__main__":
    main()