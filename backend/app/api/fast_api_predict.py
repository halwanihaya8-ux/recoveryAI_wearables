from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from app.database import DbSession
from app.recovery_ai.predict import get_last_5_days, run_prediction
from app.services import ApiKeyDep

router = APIRouter()


# ─── Manual endpoint (keep for Swagger testing) ────────────────────────────────

class PredictionInput(BaseModel):
    hrv_rmssd_ms:          list[float]
    avg_hr_day_bpm:        list[float]
    sleep_duration_hours:  list[float]
    steps:                 list[float]

    @field_validator("hrv_rmssd_ms", "avg_hr_day_bpm", "sleep_duration_hours", "steps")
    @classmethod
    def must_be_5_days(cls, v, info):
        if len(v) != 5:
            raise ValueError(f"{info.field_name} must have exactly 5 values, got {len(v)}")
        return v


@router.post("/predict")
def predict(data: PredictionInput):
    """Manual prediction — send your own 5-day arrays."""
    prediction = run_prediction(
        hrv=data.hrv_rmssd_ms,
        hr=data.avg_hr_day_bpm,
        sleep=data.sleep_duration_hours,
        steps=data.steps,
    )
    return {"prediction": round(prediction, 2)}


# ─── Automatic endpoint — pulls real data from DB ──────────────────────────────

@router.post("/predict/{user_id}")
def predict_for_user(
    user_id: UUID,
    db: DbSession,
    _api_key: ApiKeyDep,           # same auth as all other endpoints
):
    """
    Automatic prediction for a user.
    Fetches their last 5 days of HRV, HR, sleep, and steps from the database,
    then returns tomorrow's predicted HRV.
    """
    data = get_last_5_days(db, user_id)

    if data["days_found"] == 0:
        raise HTTPException(
            status_code=404,
            detail="No biometric data found for this user. "
                   "Sync at least 1 day of data first.",
        )

    prediction = run_prediction(
        hrv=data["hrv"],
        hr=data["hr"],
        sleep=data["sleep"],
        steps=data["steps"],
    )

    return {
        "user_id":    str(user_id),
        "prediction": round(prediction, 2),
        "days_used":  data["days_found"],
        "note":       "Predicted HRV (ms) for tomorrow. "
                      f"Based on {data['days_found']}/5 days of real data.",
    }