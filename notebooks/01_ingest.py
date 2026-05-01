"""
notebooks/01_ingest.py
──────────────────────
Phase 1 → Data Ingestion with schema enforcement.

Run:
    python notebooks/01_ingest.py
"""

import pandas as pd
from pathlib import Path

SCHEMA = {
    "student_id":         "string",
    "gender":             "category",
    "school_type":        "category",
    "parent_edu":         "category",
    "prior_gpa":          "float64",
    "attendance_pct":     "float64",
    "quiz_avg":           "float64",
    "assign_avg":         "float64",
    "midterm":            "float64",
    "study_hours_wk":     "float64",
    "on_time_submit_pct": "float64",
    "lms_logins_wk":      "float64",
    "forum_posts":        "float64",
    "commute_min":        "float64",
    "final_score":        "float64",
    "final_grade_band":   "category",
    "passed":             "int64",
}

csv_path = Path("data/students.csv")
if not csv_path.exists():
    raise FileNotFoundError("❌ data/students.csv not found.\n   Run: python generate_data.py first.")

print(f"📂 Loading {csv_path} …")
df = pd.read_csv(csv_path)
df = df.astype(SCHEMA)

# ── Validations ───────────────────────────────────────────────────────────────
assert df["passed"].isin([0, 1]).all(),          "passed must be 0 or 1"
assert df["attendance_pct"].between(0,100).all(),"attendance_pct out of range"
assert df["prior_gpa"].between(0, 10).all(),     "prior_gpa out of range"
assert df["final_score"].between(0, 100).all(),  "final_score out of range"
assert df["student_id"].nunique() == len(df),    "duplicate student IDs found"

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path("data/students.parquet")
df.to_parquet(out, index=False)

n_fail = (df["passed"] == 0).sum()
print(f"\n✅ Ingest complete → {out}")
print(f"   Shape        : {df.shape}")
print(f"   Pass rate    : {df['passed'].mean()*100:.1f}%")
print(f"   At-risk      : {n_fail:,} students ({n_fail/len(df)*100:.1f}%)")
print(f"   Null counts  :\n{df.isnull().sum()[df.isnull().sum()>0]}")
print(f"\n   dtypes:\n{df.dtypes}")