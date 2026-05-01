"""
notebooks/02_eda.py
───────────────────
Phase 2 → Exploratory Data Analysis

Run:
    python notebooks/02_eda.py

Saves 7 charts to outputs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
PASS_COLOR  = "#2ecc71"
FAIL_COLOR  = "#e74c3c"
BLUE        = "#3498db"
AMBER       = "#f0a500"
PALETTE     = [PASS_COLOR, FAIL_COLOR]

# ── Load ──────────────────────────────────────────────────────────────────────
print("📂 Loading data …")
df = pd.read_parquet("data/students.parquet")

passed = df[df.passed == 1]
failed = df[df.passed == 0]
print(f"   {len(df):,} students | Pass: {len(passed):,} | At-Risk/Fail: {len(failed):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Class Distribution
# ═══════════════════════════════════════════════════════════════════════════════
print("\n📊 Chart 1: Class distribution …")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Student Performance — Class Distribution", fontsize=14, fontweight="bold")

# Pass/Fail count
counts = df["passed"].value_counts()
axes[0].bar(["Passed", "At-Risk/Fail"], counts.values, color=PALETTE, edgecolor="white", linewidth=1.5)
axes[0].set_title("Pass vs At-Risk Count")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 50, f"{v:,}", ha="center", fontweight="bold")

# Pie
axes[1].pie(counts.values, labels=["Passed", "At-Risk/Fail"],
            colors=PALETTE, autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Proportion")

# Grade band
grade_counts = df["final_grade_band"].value_counts().reindex(["A","B","C","D","F"])
grade_colors = [PASS_COLOR, PASS_COLOR, AMBER, FAIL_COLOR, FAIL_COLOR]
axes[2].bar(grade_counts.index, grade_counts.values, color=grade_colors, edgecolor="white")
axes[2].set_title("Grade Band Distribution")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/01_class_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Score Distributions
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 2: Score distributions …")
score_cols = ["prior_gpa", "quiz_avg", "assign_avg", "midterm", "final_score"]
labels     = ["Prior GPA (×10)", "Quiz Avg", "Assignment Avg", "Midterm", "Final Score"]

fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle("Score Distributions — Pass vs At-Risk", fontsize=13, fontweight="bold")

for ax, col, label in zip(axes, score_cols, labels):
    scale = 10 if col == "prior_gpa" else 1
    ax.hist(passed[col] * scale,  bins=40, alpha=0.6, color=PASS_COLOR, label="Passed",     density=True)
    ax.hist(failed[col] * scale,  bins=40, alpha=0.7, color=FAIL_COLOR, label="At-Risk",    density=True)
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("Score")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_score_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/02_score_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Attendance & Study Hours
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 3: Attendance & study hours …")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Attendance & Engagement Analysis", fontsize=13, fontweight="bold")

# Attendance histogram
axes[0].hist(passed["attendance_pct"], bins=40, alpha=0.6, color=PASS_COLOR, label="Passed", density=True)
axes[0].hist(failed["attendance_pct"], bins=40, alpha=0.7, color=FAIL_COLOR, label="At-Risk", density=True)
axes[0].set_title("Attendance %")
axes[0].set_xlabel("Attendance (%)")
axes[0].legend()

# Study hours box
bp_data = [passed["study_hours_wk"], failed["study_hours_wk"]]
bp = axes[1].boxplot(bp_data, labels=["Passed", "At-Risk"], patch_artist=True, notch=True)
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[1].set_title("Study Hours / Week")
axes[1].set_ylabel("Hours")

# Scatter: attendance vs final score
scatter_sample = df.sample(2000, random_state=42)
colors = [PASS_COLOR if p else FAIL_COLOR for p in scatter_sample["passed"]]
axes[2].scatter(scatter_sample["attendance_pct"], scatter_sample["final_score"],
                c=colors, alpha=0.4, s=12)
axes[2].set_title("Attendance vs Final Score")
axes[2].set_xlabel("Attendance (%)")
axes[2].set_ylabel("Final Score")
from matplotlib.patches import Patch
axes[2].legend(handles=[Patch(color=PASS_COLOR, label="Passed"),
                         Patch(color=FAIL_COLOR, label="At-Risk")])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_attendance_study.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/03_attendance_study.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — School Type & Demographics
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 4: Demographics …")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Demographic Risk Analysis", fontsize=13, fontweight="bold")

for ax, col, title in [
    (axes[0], "school_type",  "Fail Rate by School Type"),
    (axes[1], "parent_edu",   "Fail Rate by Parent Education"),
    (axes[2], "gender",       "Fail Rate by Gender"),
]:
    fail_rate = df.groupby(col)["passed"].apply(lambda x: (x==0).mean() * 100)
    fail_rate = fail_rate.sort_values(ascending=True)
    colors = [FAIL_COLOR if v > fail_rate.mean() else PASS_COLOR for v in fail_rate.values]
    ax.barh(fail_rate.index, fail_rate.values, color=colors, edgecolor="white")
    ax.axvline(fail_rate.mean(), linestyle="--", color=AMBER, lw=1.5, label=f"Avg {fail_rate.mean():.1f}%")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("At-Risk Rate (%)")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_demographics.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/04_demographics.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5 — LMS Engagement
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 5: LMS engagement …")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("LMS Engagement vs Performance", fontsize=13, fontweight="bold")

eng_cols = ["lms_logins_wk", "on_time_submit_pct", "forum_posts"]
eng_labels = ["LMS Logins / Week", "On-Time Submit %", "Forum Posts"]

for ax, col, label in zip(axes, eng_cols, eng_labels):
    clip_val = df[col].quantile(0.98)
    ax.hist(passed[col].clip(0, clip_val), bins=40, alpha=0.6,
            color=PASS_COLOR, label="Passed", density=True)
    ax.hist(failed[col].clip(0, clip_val), bins=40, alpha=0.7,
            color=FAIL_COLOR, label="At-Risk", density=True)
    ax.set_title(label)
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_lms_engagement.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/05_lms_engagement.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 6: Correlation heatmap …")
num_cols = ["prior_gpa", "attendance_pct", "quiz_avg", "assign_avg", "midterm",
            "study_hours_wk", "on_time_submit_pct", "lms_logins_wk",
            "forum_posts", "commute_min", "passed"]

corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/06_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Risk Profile Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 7: Risk profile summary …")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("At-Risk Student Profile vs Passing Student Profile", fontsize=13, fontweight="bold")

profile_cols = ["prior_gpa", "attendance_pct", "quiz_avg",
                "assign_avg", "midterm", "study_hours_wk",
                "on_time_submit_pct", "lms_logins_wk"]
profile_labels = ["Prior GPA", "Attendance %", "Quiz Avg",
                  "Assign Avg", "Midterm", "Study Hrs/wk",
                  "On-Time %", "LMS Logins/wk"]

# Normalize each feature 0-1 for radar-style bar comparison
for ax, group, color, title in [
    (axes[0], failed, FAIL_COLOR, "At-Risk / Failing Students"),
    (axes[1], passed, PASS_COLOR, "Passing Students"),
]:
    means = [group[c].mean() for c in profile_cols]
    # Normalize relative to overall max for display
    overall_max = [df[c].max() for c in profile_cols]
    normed = [m/mx*100 for m, mx in zip(means, overall_max)]
    bars = ax.barh(profile_labels, normed, color=color, alpha=0.8, edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("% of Maximum Value")
    for bar, val in zip(bars, means):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_risk_profile.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/07_risk_profile.png")

# ── Key findings ──────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("📋 EDA Key Findings")
print("="*55)
print(f"   At-risk rate        : {(df['passed']==0).mean()*100:.1f}%")
print(f"   Avg attendance (pass): {passed['attendance_pct'].mean():.1f}%")
print(f"   Avg attendance (fail): {failed['attendance_pct'].mean():.1f}%")
print(f"   Avg study hrs (pass) : {passed['study_hours_wk'].mean():.1f} hrs/wk")
print(f"   Avg study hrs (fail) : {failed['study_hours_wk'].mean():.1f} hrs/wk")
print(f"   Avg midterm (pass)   : {passed['midterm'].mean():.1f}")
print(f"   Avg midterm (fail)   : {failed['midterm'].mean():.1f}")
print(f"\n   Charts saved to outputs/ (7 files)")
print("\n✅ EDA complete! Next → python notebooks/03_features.py")