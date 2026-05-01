"""
generate_data.py
────────────────
Generates a synthetic student performance dataset.

Run:
    python generate_data.py

Output:
    data/students.csv
    data/students.parquet

~10,000 students with realistic academic patterns.
At-risk rate is intentionally imbalanced (~30% fail/at-risk).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SEED      = 42
N_TOTAL   = 10_000
OUTPUT_DIR = Path("data")

np.random.seed(SEED)
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_students(n: int = N_TOTAL) -> pd.DataFrame:

    # ── Demographics ─────────────────────────────────────────────────────────
    genders      = np.random.choice(["M", "F", "Other"], size=n, p=[0.48, 0.48, 0.04])
    school_types = np.random.choice(["Govt", "Private", "International"], size=n, p=[0.45, 0.40, 0.15])
    parent_edu   = np.random.choice(["Below_HSC", "HSC", "UG", "PG", "PhD"], size=n,
                                     p=[0.15, 0.20, 0.35, 0.25, 0.05])

    # ── Prior GPA (scale 0–10) ────────────────────────────────────────────────
    # International school students tend to have higher prior GPA
    prior_gpa = np.where(
        school_types == "International",
        np.random.normal(7.8, 1.0, n),
        np.where(school_types == "Private",
                 np.random.normal(7.0, 1.3, n),
                 np.random.normal(6.2, 1.5, n))
    ).clip(2.0, 10.0)

    # ── Attendance % ─────────────────────────────────────────────────────────
    attendance_pct = np.random.beta(8, 2, n) * 100   # skewed right (most attend well)
    attendance_pct = attendance_pct.clip(20, 100)

    # ── Weekly study hours ────────────────────────────────────────────────────
    study_hours_wk = np.random.lognormal(mean=2.1, sigma=0.5, size=n).clip(0.5, 20)

    # ── Quiz average (0–100) ─────────────────────────────────────────────────
    quiz_avg = (0.4 * prior_gpa * 10 +
                0.3 * attendance_pct * 0.8 +
                0.3 * study_hours_wk * 4 +
                np.random.normal(0, 8, n)).clip(10, 100)

    # ── Assignment average ────────────────────────────────────────────────────
    assign_avg = (0.45 * prior_gpa * 10 +
                  0.25 * attendance_pct * 0.7 +
                  0.30 * study_hours_wk * 3.5 +
                  np.random.normal(0, 7, n)).clip(10, 100)

    # ── Midterm score ─────────────────────────────────────────────────────────
    midterm = (0.40 * prior_gpa * 10 +
               0.25 * quiz_avg * 0.9 +
               0.20 * assign_avg * 0.8 +
               0.15 * study_hours_wk * 3 +
               np.random.normal(0, 9, n)).clip(5, 100)

    # ── LMS engagement ────────────────────────────────────────────────────────
    lms_logins_wk       = np.random.poisson(lam=5, size=n).clip(0, 30).astype(float)
    on_time_submit_pct  = (0.5 * assign_avg + 0.3 * attendance_pct * 0.5 +
                           np.random.normal(0, 10, n)).clip(0, 100)
    forum_posts         = np.random.poisson(lam=2, size=n).clip(0, 30).astype(float)

    # ── Commute time (minutes) ────────────────────────────────────────────────
    commute_min = np.random.lognormal(mean=3.3, sigma=0.6, size=n).clip(5, 180)

    # ── Final score (target-generating process) ───────────────────────────────
    final_score = (
        0.25 * midterm +
        0.20 * quiz_avg +
        0.20 * assign_avg +
        0.15 * prior_gpa * 10 +
        0.10 * attendance_pct * 0.8 +
        0.05 * study_hours_wk * 3 +
        0.03 * on_time_submit_pct * 0.5 +
        0.02 * lms_logins_wk * 2 +
        np.random.normal(0, 5, n)
    ).clip(0, 100)

    # ── Grade band ────────────────────────────────────────────────────────────
    def to_grade(s):
        if s >= 80:   return "A"
        elif s >= 65: return "B"
        elif s >= 50: return "C"
        elif s >= 35: return "D"
        else:         return "F"

    final_grade_band = np.array([to_grade(s) for s in final_score])
    passed           = (final_score >= 35).astype(int)

    # ── Student IDs ───────────────────────────────────────────────────────────
    student_ids = [f"STU{i:06d}" for i in range(n)]

    df = pd.DataFrame({
        "student_id":          student_ids,
        "gender":              genders,
        "school_type":         school_types,
        "parent_edu":          parent_edu,
        "prior_gpa":           prior_gpa.round(2),
        "attendance_pct":      attendance_pct.round(2),
        "quiz_avg":            quiz_avg.round(2),
        "assign_avg":          assign_avg.round(2),
        "midterm":             midterm.round(2),
        "study_hours_wk":      study_hours_wk.round(2),
        "on_time_submit_pct":  on_time_submit_pct.round(2),
        "lms_logins_wk":       lms_logins_wk,
        "forum_posts":         forum_posts,
        "commute_min":         commute_min.round(1),
        "final_score":         final_score.round(2),
        "final_grade_band":    final_grade_band,
        "passed":              passed,
    })

    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ── Schema enforcement ────────────────────────────────────────────────────────
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


if __name__ == "__main__":
    print("🔧 Generating synthetic student data …")
    df = generate_students()

    # Save CSV
    csv_path = OUTPUT_DIR / "students.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved   → {csv_path}  ({len(df):,} rows)")

    # Save typed Parquet
    df_typed = df.astype(SCHEMA)
    pq_path  = OUTPUT_DIR / "students.parquet"
    df_typed.to_parquet(pq_path, index=False)
    print(f"✅ Parquet saved → {pq_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_fail = (df["passed"] == 0).sum()
    print(f"\n📊 Dataset Summary")
    print(f"   Total students  : {len(df):,}")
    print(f"   Passed          : {df['passed'].sum():,}  ({df['passed'].mean()*100:.1f}%)")
    print(f"   Failed/At-Risk  : {n_fail:,}  ({n_fail/len(df)*100:.1f}%)")
    print(f"\n   Grade Distribution:")
    print(df["final_grade_band"].value_counts().sort_index().to_string())
    print(f"\n   Avg Final Score : {df['final_score'].mean():.2f}")
    print(f"   Avg Attendance  : {df['attendance_pct'].mean():.1f}%")
    print(f"   Avg Study Hours : {df['study_hours_wk'].mean():.1f} hrs/week")
    print(f"\n   School Type breakdown:")
    print(df["school_type"].value_counts().to_string())