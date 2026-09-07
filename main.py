"""Train an AquaG water-logging model from the supplied Delhi flood dataset.

The supplied archive is a page-by-page export of the Delhi Flood Control Order.
Its most consistently structured records are the water-logging tables on pages
58-83, so those pages are normalized into ``zone_flood_data.csv``.

The model target is based on measured location/date observations, not on the
number of dates attached to a location. A positive row is a location/date
reported as water-logged by the order. A matched negative row is a date in the
same source year for which that location is absent from the order's
water-logging list. The saved metadata records this source-observation
definition explicitly: an absent incident record is not proof that a location
could never flood outside the source's observation coverage.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "flood_data_csv_by_page"
DATASET_PATH = ROOT / "zone_flood_data.csv"
MODEL_PATH = ROOT / "aquag_model.pkl"
VERSIONS_PATH = ROOT / "package_versions.txt"
INCIDENT_PAGE_RANGE = range(58, 84)
DATE_PATTERN = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")

FEATURE_COLUMNS = [
    "section_year",
    "page_number",
    "event_month",
    "event_day",
    "road_length",
    "location_length",
    "has_underpass",
    "has_drain",
    "has_bridge",
]


def clean_text(value: str) -> str:
    """Collapse line breaks and repeated whitespace from a CSV cell."""

    return re.sub(r"\s+", " ", value or "").strip()


def extract_dates(*values: str) -> list[str]:
    """Return ISO dates found in one or more source cells."""

    dates = []
    for value in values:
        for day, month, year in DATE_PATTERN.findall(value or ""):
            dates.append(f"{int(year):04d}-{int(month):02d}-{int(day):02d}")
    return dates


def is_record_number(value: str) -> bool:
    """Recognize the serial-number cells used by the source tables."""

    return bool(re.fullmatch(r"\s*(?:\d+|[ivxIVX]+)[.)]?\s*", value or ""))


def page_number(path: Path) -> int:
    match = re.search(r"page_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not determine page number from {path.name}")
    return int(match.group(1))


def load_water_logging_records() -> list[dict[str, object]]:
    """Parse the water-logging tables while preserving source page metadata."""

    records: list[dict[str, object]] = []
    for number in INCIDENT_PAGE_RANGE:
        path = RAW_DIR / f"page_{number:03d}.csv"
        if not path.exists():
            continue

        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))

        title = clean_text(rows[0][0]) if rows and rows[0] else ""
        year_match = re.search(r"\b(20\d{2})\b", title)
        if not year_match:
            continue
        section_year = int(year_match.group(1))

        current: dict[str, object] | None = None
        for row in rows[1:]:
            values = [clean_text(cell) for cell in row]
            if not any(values):
                continue

            if is_record_number(values[0]):
                if current is not None:
                    records.append(current)
                current = {
                    "source_page": number,
                    "section_year": section_year,
                    "row_number": values[0].rstrip("."),
                    "road": values[1] if len(values) > 1 else "",
                    "location": values[2] if len(values) > 2 else "",
                    "date_cells": [
                        cell for cell in values[3:] if cell
                    ],
                }
            elif current is not None:
                # Wrapped CSV cells and continuation pages sometimes put the
                # remaining date in a row with no serial number.
                current["date_cells"].extend(cell for cell in values if cell)

        if current is not None:
            records.append(current)

    normalized: list[dict[str, object]] = []
    for record in records:
        dates = extract_dates(*record["date_cells"])
        if not dates:
            continue
        event_date = dates[0]
        normalized.append(
            {
                "source_page": record["source_page"],
                "section_year": record["section_year"],
                "row_number": record["row_number"],
                "road": record["road"],
                "location": record["location"],
                "date_values": "|".join(dates),
                "event_date": event_date,
                "event_year": int(event_date[:4]),
                "event_month": int(event_date[5:7]),
                "event_day": int(event_date[8:10]),
                "event_count": len(dates),
            }
        )
    return normalized


def build_dataset(records: list[dict[str, object]]) -> pd.DataFrame:
    """Build measured location/date outcomes and model-ready features.

    The source has positive observations only, so each positive observation is
    paired with one deterministic, same-year date that is absent for that
    location. These controls represent the measurable target available in the
    order: incident listed versus incident not listed on that date. They never
    use ``event_count`` to define the target.
    """

    if not records:
        raise RuntimeError(
            f"No water-logging records found in {RAW_DIR}. "
            "Make sure the supplied page CSVs are present."
        )

    source_frame = pd.DataFrame(records)
    source_frame["observed_dates"] = source_frame["date_values"].str.split("|")
    all_dates_by_year: dict[int, list[str]] = {}
    for dates in source_frame["observed_dates"]:
        for date in dates:
            all_dates_by_year.setdefault(int(date[:4]), set()).add(date)
    all_dates_by_year = {
        year: sorted(dates) for year, dates in all_dates_by_year.items()
    }

    measured_rows: list[dict[str, object]] = []
    for source_row in source_frame.to_dict("records"):
        observed_dates = list(source_row["observed_dates"])
        observed_set = set(observed_dates)
        for event_date in observed_dates:
            measured_rows.append(
                {
                    **{
                        key: value
                        for key, value in source_row.items()
                        if key != "observed_dates"
                    },
                    "event_date": event_date,
                    "event_year": int(event_date[:4]),
                    "event_month": int(event_date[5:7]),
                    "event_day": int(event_date[8:10]),
                    "outcome_source": (
                        "Delhi Flood Control Order water-logging incident list"
                    ),
                    "flood_outcome": 1,
                }
            )

        # Use the first available same-year non-incident date as a matched
        # control. Sorting makes the generated dataset reproducible.
        for event_date in observed_dates:
            year_dates = all_dates_by_year[int(event_date[:4])]
            control_dates = [
                date for date in year_dates if date not in observed_set
            ]
            if not control_dates:
                continue
            control_date = control_dates[0]
            measured_rows.append(
                {
                    **{
                        key: value
                        for key, value in source_row.items()
                        if key != "observed_dates"
                    },
                    "event_date": control_date,
                    "event_year": int(control_date[:4]),
                    "event_month": int(control_date[5:7]),
                    "event_day": int(control_date[8:10]),
                    "event_count": 0,
                    "outcome_source": (
                        "Delhi Flood Control Order: location absent from "
                        "water-logging incident list on this date"
                    ),
                    "flood_outcome": 0,
                }
            )

    frame = pd.DataFrame(measured_rows)
    if frame.empty or frame["flood_outcome"].nunique() < 2:
        raise RuntimeError(
            "The source did not produce both measured incident and matched "
            "non-incident outcomes."
        )
    frame["label"] = frame["flood_outcome"].astype(int)
    combined_text = (frame["road"] + " " + frame["location"]).str.lower()
    frame["road_length"] = frame["road"].str.len()
    frame["location_length"] = frame["location"].str.len()
    frame["page_number"] = frame["source_page"]
    frame["has_underpass"] = combined_text.str.contains("underpass", regex=False).astype(int)
    frame["has_drain"] = combined_text.str.contains("drain", regex=False).astype(int)
    frame["has_bridge"] = combined_text.str.contains("bridge", regex=False).astype(int)
    return frame


def train_model(
    frame: pd.DataFrame,
) -> tuple[XGBClassifier, dict[str, object]]:
    """Train and evaluate against the measured flood outcome."""

    if frame["flood_outcome"].nunique() < 2:
        raise RuntimeError(
            "The labeled dataset contains only one measured outcome class; "
            "a classifier needs both incident and non-incident records."
        )

    X = frame[FEATURE_COLUMNS].astype(float)
    y = frame["flood_outcome"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "test_rows": int(len(y_test)),
    }
    return model, metrics


def save_outputs(
    frame: pd.DataFrame,
    model: XGBClassifier,
    metrics: dict[str, object],
) -> None:
    """Write the requested dataset, model, and package version record."""

    output_columns = [
        "source_page",
        "section_year",
        "row_number",
        "road",
        "location",
        "date_values",
        "event_date",
        "event_year",
        "event_month",
        "event_day",
        "event_count",
        "outcome_source",
        "flood_outcome",
        "label",
    ]
    frame[output_columns].to_csv(DATASET_PATH, index=False)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": "flood_outcome",
            "label_definition": (
                "1 when the Delhi Flood Control Order lists water-logging "
                "for the location/date; 0 when the location is absent from "
                "the order's water-logging list on a matched same-year date"
            ),
            "target_source": (
                "Delhi Flood Control Order, water-logging tables on pages 58-83"
            ),
            "target_caveat": (
                "A negative is a verified non-incident in this source list, "
                "not proof of no flooding outside the order's observation "
                "coverage."
            ),
            "metrics": metrics,
            "accuracy": metrics["accuracy"],
            "class_counts": {
                str(key): int(value)
                for key, value in frame["flood_outcome"]
                .value_counts()
                .sort_index()
                .items()
            },
            "training_rows": int(len(frame)),
        },
        MODEL_PATH,
    )
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


def main() -> None:
    records = load_water_logging_records()
    frame = build_dataset(records)
    model, metrics = train_model(frame)
    save_outputs(frame, model, metrics)
    print(f"Records loaded: {len(frame)}")
    print(
        "Measured outcome counts: "
        f"{frame['flood_outcome'].value_counts().sort_index().to_dict()}"
    )
    print(
        "Metrics: "
        + ", ".join(
            f"{name}={value:.4f}"
            for name, value in metrics.items()
            if isinstance(value, float)
        )
    )
    print(f"Saved dataset: {DATASET_PATH.name}")
    print(f"Saved model: {MODEL_PATH.name}")
    print(f"Saved versions: {VERSIONS_PATH.name}")


if __name__ == "__main__":
    main()
