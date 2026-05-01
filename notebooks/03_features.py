"""
notebooks/03_features.py
─────────────────────────
Phase 2 → Feature Engineering + Train/Test Split

What it does:
    - Engineers 15+ new features from raw columns
    - Builds scikit-learn preprocessing pipeline
    - Stratified train/test split (no leakage — excludes final_score)
    - Saves processed splits + pipeline to data/

Run:
    python notebooks/03_features.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20

# ── Load ──────────────────────────────────────────────────────────────────────
print("📂 Loading students.parquet …")
df = pd.read_parquet(DATA_DIR / "students.parquet")
print(f"   {len(df):,} rows loaded")

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🔧 Engineering features …")

# ── Academic performance composites ──────────────────────────────────────────
df["academic_avg"]        = (df["quiz_avg"] + df["assign_avg"]) / 2
df["theory_vs_practice"]  = df["midterm"] - df["academic_avg"]
df["gpa_scaled"]          = df["prior_gpa"] * 10          # align to 0–100 scale
df["overall_academic"]    = (df["gpa_scaled"] * 0.3 +
                              df["quiz_avg"]   * 0.25 +
                              df["assign_avg"] * 0.25 +
                              df["midterm"]    * 0.20)

# ── Engagement score ──────────────────────────────────────────────────────────
df["engagement_score"]    = (df["lms_logins_wk"]      * 3 +
                              df["on_time_submit_pct"] * 0.5 +
                              df["forum_posts"]        * 2 +
                              df["attendance_pct"]     * 0.3).clip(0, 100)

# ── Risk flags ────────────────────────────────────────────────────────────────
df["low_attendance"]      = (df["attendance_pct"] < 60).astype(int)
df["low_study_hours"]     = (df["study_hours_wk"] < 3).astype(int)
df["weak_midterm"]        = (df["midterm"] < 40).astype(int)
df["low_quiz"]            = (df["quiz_avg"] < 40).astype(int)
df["low_lms"]             = (df["lms_logins_wk"] < 2).astype(int)
df["risk_flag_sum"]       = (df["low_attendance"] + df["low_study_hours"] +
                              df["weak_midterm"]   + df["low_quiz"] + df["low_lms"])

# ── Interaction features ──────────────────────────────────────────────────────
df["attendance_x_study"]  = df["attendance_pct"] * df["study_hours_wk"] / 100
df["midterm_x_quiz"]      = df["midterm"] * df["quiz_avg"] / 100
df["commute_burden"]      = (df["commute_min"] > 60).astype(int)

print(f"   Engineered features added: 14")

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE LIST (EXCLUDE LEAKAGE COLUMNS)
# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTANT: final_score and final_grade_band are EXCLUDED — they are derived
# from the target and would cause data leakage.

NUMERIC_FEATURES = [
    # Raw
    "prior_gpa", "attendance_pct", "quiz_avg", "assign_avg", "midterm",
    "study_hours_wk", "on_time_submit_pct", "lms_logins_wk",
    "forum_posts", "commute_min",
    # Engineered
    "academic_avg", "theory_vs_practice", "gpa_scaled", "overall_academic",
    "engagement_score", "attendance_x_study", "midterm_x_quiz",
    "low_attendance", "low_study_hours", "weak_midterm", "low_quiz",
    "low_lms", "risk_flag_sum", "commute_burden",
]

CATEGORICAL_FEATURES = ["gender", "school_type", "parent_edu"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET       = "passed"

print(f"   Total features : {len(ALL_FEATURES)}  ({len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical)")

# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🔧 Building sklearn preprocessing pipeline …")

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, NUMERIC_FEATURES),
    ("cat", cat_pipe, CATEGORICAL_FEATURES),
])

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT (stratified)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n✂️  Stratified train/test split …")

X = df[ALL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print(f"   Train : {len(X_train):,}  | Pass={y_train.sum():,}  Fail={(y_train==0).sum():,}  ({y_train.mean()*100:.1f}% pass)")
print(f"   Test  : {len(X_test):,}   | Pass={y_test.sum():,}   Fail={(y_test==0).sum():,}   ({y_test.mean()*100:.1f}% pass)")

# ═══════════════════════════════════════════════════════════════════════════════
# FIT PREPROCESSOR ON TRAIN SET ONLY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n⚙️  Fitting preprocessor on training set …")
preprocessor.fit(X_train)

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n💾 Saving splits and pipeline …")

X_train.to_parquet(DATA_DIR / "X_train.parquet", index=False)
X_test.to_parquet(DATA_DIR  / "X_test.parquet",  index=False)
y_train.to_frame().to_parquet(DATA_DIR / "y_train.parquet", index=False)
y_test.to_frame().to_parquet(DATA_DIR  / "y_test.parquet",  index=False)

joblib.dump(preprocessor,       MODEL_DIR / "preprocessor.joblib")
joblib.dump(ALL_FEATURES,       DATA_DIR  / "feature_list.joblib")
joblib.dump(NUMERIC_FEATURES,   DATA_DIR  / "numeric_features.joblib")
joblib.dump(CATEGORICAL_FEATURES, DATA_DIR / "categorical_features.joblib")

print(f"   ✅ data/X_train.parquet  ({len(X_train):,} rows)")
print(f"   ✅ data/X_test.parquet   ({len(X_test):,} rows)")
print(f"   ✅ data/y_train.parquet")
print(f"   ✅ data/y_test.parquet")
print(f"   ✅ models/preprocessor.joblib")
print(f"   ✅ data/feature_list.joblib")

print("\n" + "="*55)
print("✅ Feature engineering complete!")
print("   Next → python notebooks/04_train.py")
print("="*55)