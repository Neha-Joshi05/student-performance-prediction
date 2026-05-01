"""
serving/api.py
──────────────
Phase 4 → FastAPI Student Performance Prediction API

Endpoints:
    GET  /              → health check
    GET  /metrics       → model performance metrics
    POST /predict       → single student prediction
    POST /predict/batch → batch predictions (up to 500)

Run:
    uvicorn serving.api:app --reload --port 8000
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Load artifacts ─────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "data"

model        = joblib.load(MODEL_DIR / "xgb_student_model.joblib")
threshold    = joblib.load(MODEL_DIR / "optimal_threshold.joblib")
metrics      = joblib.load(MODEL_DIR / "eval_metrics.joblib")
preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
features     = joblib.load(DATA_DIR  / "feature_list.joblib")
num_feats    = joblib.load(DATA_DIR  / "numeric_features.joblib")
cat_feats    = joblib.load(DATA_DIR  / "categorical_features.joblib")

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Student Performance Prediction API",
    description = "Real-time student at-risk prediction powered by XGBoost + SHAP",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────
class StudentInput(BaseModel):
    gender:             str   = Field(..., description="M / F / Other")
    school_type:        str   = Field(..., description="Govt / Private / International")
    parent_edu:         str   = Field(..., description="Below_HSC / HSC / UG / PG / PhD")
    prior_gpa:          float = Field(..., ge=0, le=10,  description="GPA on 0–10 scale")
    attendance_pct:     float = Field(..., ge=0, le=100, description="Attendance percentage")
    quiz_avg:           float = Field(..., ge=0, le=100)
    assign_avg:         float = Field(..., ge=0, le=100)
    midterm:            float = Field(..., ge=0, le=100)
    study_hours_wk:     float = Field(..., ge=0, le=20,  description="Study hours per week")
    on_time_submit_pct: float = Field(75.0, ge=0, le=100)
    lms_logins_wk:      float = Field(5.0,  ge=0)
    forum_posts:        float = Field(2.0,  ge=0)
    commute_min:        float = Field(30.0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "M", "school_type": "Govt", "parent_edu": "HSC",
                "prior_gpa": 5.2, "attendance_pct": 55.0,
                "quiz_avg": 38.0, "assign_avg": 42.0, "midterm": 36.0,
                "study_hours_wk": 2.0, "on_time_submit_pct": 50.0,
                "lms_logins_wk": 1.0, "forum_posts": 0.0, "commute_min": 90.0,
            }
        }


class PredictionResponse(BaseModel):
    pass_probability:   float
    is_at_risk:         bool
    risk_level:         str
    risk_score:         float
    threshold_used:     float
    interventions:      list[str]
    timestamp:          str


class BatchRequest(BaseModel):
    students: list[StudentInput] = Field(..., max_items=500)


class BatchResponse(BaseModel):
    total:          int
    at_risk_count:  int
    at_risk_pct:    float
    results:        list[PredictionResponse]


# ── Feature engineering (mirrors notebooks/03_features.py) ────────────────────
def engineer_features(s: StudentInput) -> pd.DataFrame:
    row = s.dict()

    gpa_scaled          = row["prior_gpa"] * 10
    academic_avg        = (row["quiz_avg"] + row["assign_avg"]) / 2
    theory_vs_practice  = row["midterm"] - academic_avg
    overall_academic    = (gpa_scaled * 0.3 + row["quiz_avg"] * 0.25 +
                           row["assign_avg"] * 0.25 + row["midterm"] * 0.20)
    engagement_score    = min(100, (row["lms_logins_wk"] * 3 +
                                    row["on_time_submit_pct"] * 0.5 +
                                    row["forum_posts"] * 2 +
                                    row["attendance_pct"] * 0.3))

    low_attendance     = int(row["attendance_pct"] < 60)
    low_study_hours    = int(row["study_hours_wk"] < 3)
    weak_midterm       = int(row["midterm"] < 40)
    low_quiz           = int(row["quiz_avg"] < 40)
    low_lms            = int(row["lms_logins_wk"] < 2)
    risk_flag_sum      = low_attendance + low_study_hours + weak_midterm + low_quiz + low_lms
    attendance_x_study = row["attendance_pct"] * row["study_hours_wk"] / 100
    midterm_x_quiz     = row["midterm"] * row["quiz_avg"] / 100
    commute_burden     = int(row["commute_min"] > 60)

    engineered = {
        "gender":             row["gender"],
        "school_type":        row["school_type"],
        "parent_edu":         row["parent_edu"],
        "prior_gpa":          row["prior_gpa"],
        "attendance_pct":     row["attendance_pct"],
        "quiz_avg":           row["quiz_avg"],
        "assign_avg":         row["assign_avg"],
        "midterm":            row["midterm"],
        "study_hours_wk":     row["study_hours_wk"],
        "on_time_submit_pct": row["on_time_submit_pct"],
        "lms_logins_wk":      row["lms_logins_wk"],
        "forum_posts":        row["forum_posts"],
        "commute_min":        row["commute_min"],
        "academic_avg":       academic_avg,
        "theory_vs_practice": theory_vs_practice,
        "gpa_scaled":         gpa_scaled,
        "overall_academic":   overall_academic,
        "engagement_score":   engagement_score,
        "low_attendance":     low_attendance,
        "low_study_hours":    low_study_hours,
        "weak_midterm":       weak_midterm,
        "low_quiz":           low_quiz,
        "low_lms":            low_lms,
        "risk_flag_sum":      risk_flag_sum,
        "attendance_x_study": attendance_x_study,
        "midterm_x_quiz":     midterm_x_quiz,
        "commute_burden":     commute_burden,
    }
    return pd.DataFrame([engineered])[features]


def get_risk_level(pass_prob: float) -> str:
    if pass_prob < 0.30:  return "CRITICAL"
    if pass_prob < 0.50:  return "HIGH"
    if pass_prob < 0.70:  return "MEDIUM"
    return "LOW"


def get_interventions(s: StudentInput, pass_prob: float) -> list[str]:
    tips = []
    if s.attendance_pct < 60:
        tips.append("⚠️ Attendance below 60% — prioritise class presence immediately")
    if s.study_hours_wk < 3:
        tips.append("📚 Increase study hours to at least 5–7 hrs/week")
    if s.midterm < 40:
        tips.append("📝 Midterm score is weak — schedule remedial sessions")
    if s.quiz_avg < 40:
        tips.append("🧠 Low quiz average — review core concepts weekly")
    if s.lms_logins_wk < 2:
        tips.append("💻 Increase LMS engagement — log in at least 4× per week")
    if s.on_time_submit_pct < 60:
        tips.append("📅 Improve assignment submission rate — aim for 80%+")
    if s.commute_min > 90:
        tips.append("🚌 Long commute detected — consider online resources to save time")
    if not tips:
        if pass_prob < 0.70:
            tips.append("📈 Performance is borderline — maintain current effort and seek feedback")
        else:
            tips.append("✅ Student is on track — keep up the good work!")
    return tips[:4]


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "status":    "online",
        "service":   "Student Performance Prediction API",
        "version":   "1.0.0",
        "threshold": round(threshold, 3),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics", tags=["Model"])
def get_metrics():
    return {
        "model":     "XGBoost (Optuna-tuned)",
        "threshold": round(threshold, 3),
        "metrics": {
            "roc_auc":   round(metrics["roc_auc"],   4),
            "pr_auc":    round(metrics["pr_auc"],    4),
            "precision": round(metrics["precision"], 4),
            "recall":    round(metrics["recall"],    4),
            "f1_score":  round(metrics["f1"],        4),
        },
        "features": len(features),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(student: StudentInput):
    try:
        df_raw      = engineer_features(student)
        df_proc     = preprocessor.transform(df_raw)
        pass_prob   = float(model.predict_proba(df_proc)[0, 1])
        is_at_risk  = pass_prob < threshold
        risk_score  = round(1 - pass_prob, 4)

        return PredictionResponse(
            pass_probability = round(pass_prob, 4),
            is_at_risk       = is_at_risk,
            risk_level       = get_risk_level(pass_prob),
            risk_score       = risk_score,
            threshold_used   = round(threshold, 3),
            interventions    = get_interventions(student, pass_prob),
            timestamp        = datetime.utcnow().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(req: BatchRequest):
    try:
        results = []
        for student in req.students:
            df_raw    = engineer_features(student)
            df_proc   = preprocessor.transform(df_raw)
            pass_prob = float(model.predict_proba(df_proc)[0, 1])
            is_at_risk = pass_prob < threshold
            results.append(PredictionResponse(
                pass_probability = round(pass_prob, 4),
                is_at_risk       = is_at_risk,
                risk_level       = get_risk_level(pass_prob),
                risk_score       = round(1 - pass_prob, 4),
                threshold_used   = round(threshold, 3),
                interventions    = get_interventions(student, pass_prob),
                timestamp        = datetime.utcnow().isoformat(),
            ))

        at_risk = sum(r.is_at_risk for r in results)
        return BatchResponse(
            total         = len(results),
            at_risk_count = at_risk,
            at_risk_pct   = round(at_risk / len(results) * 100, 2),
            results       = results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.api:app", host="0.0.0.0", port=8000, reload=True)