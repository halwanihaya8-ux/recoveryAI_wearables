import numpy as np
import joblib
import os
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date

BASE_DIR = os.path.dirname(__file__)
model         = joblib.load(os.path.join(BASE_DIR, "modelrecovery", "model.pkl"))
scaler        = joblib.load(os.path.join(BASE_DIR, "modelrecovery", "scaler.pkl"))
feature_index = joblib.load(os.path.join(BASE_DIR, "modelrecovery", "feature_index.pkl"))


def engineer_features(hrv, hr, sleep, steps) -> np.ndarray:
    hrv   = np.array(hrv,   dtype=float)
    hr    = np.array(hr,    dtype=float)
    sleep = np.array(sleep, dtype=float)
    steps = np.array(steps, dtype=float)

    hrv_z   = hrv   - np.median(hrv)
    hr_z    = hr    - np.median(hr)
    sleep_z = sleep - np.median(sleep)

    strain_ewm = np.zeros(5)
    strain_ewm[0] = steps[0]
    for i in range(1, 5):
        strain_ewm[i] = 0.5 * steps[i] + 0.5 * strain_ewm[i - 1]

    sleep_debt = np.array([np.mean(sleep[:i+1]) - sleep[i] for i in range(5)])

    rolling3 = np.array([np.mean(hrv[max(0, i-2):i+1]) for i in range(5)])
    hrv_trend = np.zeros(5)
    hrv_trend[1:] = np.diff(rolling3)

    recovery_ratio = hrv / (hr + 1)

    rows = np.stack(
        [hrv_z, hr_z, sleep_z, strain_ewm, sleep_debt, hrv_trend, recovery_ratio],
        axis=1
    )
    return rows.flatten()


def run_prediction(hrv, hr, sleep, steps) -> float:
    X          = engineer_features(hrv, hr, sleep, steps).reshape(1, -1)
    X_scaled   = scaler.transform(X)
    X_selected = X_scaled[:, feature_index]
    return float(model.predict(X_selected)[0])


def get_last_5_days(db: Session, user_id: UUID) -> dict:
    from app.models.data_point_series import DataPointSeries
    from app.models.data_source import DataSource
    from app.models.event_record import EventRecord
    from app.schemas.enums import SeriesType, get_series_type_id

    hrv_id   = get_series_type_id(SeriesType.heart_rate_variability_rmssd)
    hr_id    = get_series_type_id(SeriesType.heart_rate)
    steps_id = get_series_type_id(SeriesType.steps)

    # --- Find the most recent data point for this user ---
    # Anchor the window here instead of using datetime.now()
    # This works regardless of when the data was seeded
    latest_row = (
        db.query(func.max(DataPointSeries.recorded_at))
        .join(DataSource, DataPointSeries.data_source_id == DataSource.id)
        .filter(DataSource.user_id == user_id)
        .scalar()
    )

    if latest_row is None:
        return {"hrv": [], "hr": [], "sleep": [], "steps": [], "days_found": 0}

    end   = latest_row
    start = end - timedelta(days=6)

    hrv_id   = get_series_type_id(SeriesType.heart_rate_variability_rmssd)
    hr_id    = get_series_type_id(SeriesType.heart_rate)
    steps_id = get_series_type_id(SeriesType.steps)

    rows = (
        db.query(
            cast(DataPointSeries.recorded_at, Date).label("day"),
            DataPointSeries.series_type_definition_id.label("type_id"),
            func.avg(DataPointSeries.value).label("avg_val"),
        )
        .join(DataSource, DataPointSeries.data_source_id == DataSource.id)
        .filter(
            DataSource.user_id == user_id,
            DataPointSeries.recorded_at >= start,
            DataPointSeries.recorded_at <= end,
            DataPointSeries.series_type_definition_id.in_([hrv_id, hr_id, steps_id]),
        )
        .group_by("day", DataPointSeries.series_type_definition_id)
        .order_by("day")
        .all()
    )

    daily: dict[str, dict] = {}
    for row in rows:
        key = str(row.day)
        if key not in daily:
            daily[key] = {}
        if row.type_id == hrv_id:
            daily[key]["hrv"] = float(row.avg_val)
        elif row.type_id == hr_id:
            daily[key]["hr"] = float(row.avg_val)
        elif row.type_id == steps_id:
            daily[key]["steps"] = float(row.avg_val)

    sleep_rows = (
        db.query(
            cast(EventRecord.start_datetime, Date).label("day"),
            func.avg(EventRecord.duration_seconds).label("avg_dur"),
        )
        .join(DataSource, EventRecord.data_source_id == DataSource.id)
        .filter(
            DataSource.user_id == user_id,
            EventRecord.category == "sleep",
            EventRecord.start_datetime >= start,
            EventRecord.start_datetime <= end,
        )
        .group_by("day")
        .order_by("day")
        .all()
    )

    for row in sleep_rows:
        key = str(row.day)
        if key not in daily:
            daily[key] = {}
        daily[key]["sleep"] = float(row.avg_dur) / 3600.0

    sorted_days = sorted(daily.keys())[-5:]

    DEFAULT_HRV, DEFAULT_HR, DEFAULT_SLEEP, DEFAULT_STEPS = 55.0, 65.0, 7.0, 7500.0

    hrv_list   = [daily[d].get("hrv",   DEFAULT_HRV)   for d in sorted_days]
    hr_list    = [daily[d].get("hr",    DEFAULT_HR)    for d in sorted_days]
    sleep_list = [daily[d].get("sleep", DEFAULT_SLEEP) for d in sorted_days]
    steps_list = [daily[d].get("steps", DEFAULT_STEPS) for d in sorted_days]

    while len(hrv_list) < 5:
        hrv_list.insert(0, DEFAULT_HRV)
        hr_list.insert(0, DEFAULT_HR)
        sleep_list.insert(0, DEFAULT_SLEEP)
        steps_list.insert(0, DEFAULT_STEPS)

    return {
        "hrv": hrv_list, "hr": hr_list,
        "sleep": sleep_list, "steps": steps_list,
        "days_found": len(sorted_days),
    }