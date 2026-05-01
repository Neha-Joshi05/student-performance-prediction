"""
notebooks/05_explain.py
────────────────────────
Phase 3 → SHAP Explainability

Run:
    python notebooks/05_explain.py
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR   = Path("data")
MODEL_DIR  = Path("models")
OUTPUT_DIR = Path("outputs")

PASS_COLOR = "#2ecc71"
FAIL_COLOR = "#e74c3c"

# ── Load ──────────────────────────────────────────────────────────────────────
print("📂 Loading model and test data …")
model        = joblib.load(MODEL_DIR / "xgb_student_model.joblib")
threshold    = joblib.load(MODEL_DIR / "optimal_threshold.joblib")
preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
num_feats    = joblib.load(DATA_DIR  / "numeric_features.joblib")
cat_feats    = joblib.load(DATA_DIR  / "categorical_features.joblib")

X_test_raw = pd.read_parquet(DATA_DIR / "X_test.parquet")
y_test     = pd.read_parquet(DATA_DIR / "y_test.parquet").squeeze()

# Reconstructed feature names after OHE
ohe       = preprocessor.named_transformers_["cat"]["ohe"]
cat_names = list(ohe.get_feature_names_out(cat_feats))
feat_names = num_feats + cat_names

X_test = preprocessor.transform(X_test_raw)
X_test_df = pd.DataFrame(X_test, columns=feat_names)

print(f"   Test set : {X_test_df.shape}  |  threshold={threshold:.3f}")

# ── Sample for SHAP ───────────────────────────────────────────────────────────
sample_idx = np.random.RandomState(42).choice(len(X_test_df), size=min(1500, len(X_test_df)), replace=False)
shap_sample    = X_test_df.iloc[sample_idx]
y_test_sample  = y_test.iloc[sample_idx]
probs_sample   = model.predict_proba(shap_sample)[:, 1]

print(f"   SHAP sample : {len(shap_sample)} rows")

# ── Compute SHAP ──────────────────────────────────────────────────────────────
print("\n🔍 Computing SHAP values …")
explainer   = shap.TreeExplainer(model)
shap_values = explainer(shap_sample)
print("   ✅ SHAP values computed")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Beeswarm
# ═══════════════════════════════════════════════════════════════════════════════
print("\n📊 Chart 1: SHAP beeswarm …")
plt.figure(figsize=(10, 9))
shap.summary_plot(shap_values, shap_sample, show=False, max_display=20, plot_size=None)
plt.title("SHAP Feature Impact — Student Performance", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "14_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/14_shap_beeswarm.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Bar
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 2: SHAP bar …")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, shap_sample, plot_type="bar",
                  show=False, max_display=20, plot_size=None)
plt.title("Mean |SHAP| Feature Importance", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "15_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/15_shap_bar.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Dependence plots (top 4)
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 3: SHAP dependence …")
mean_shap  = np.abs(shap_values.values).mean(axis=0)
top4_idx   = np.argsort(mean_shap)[::-1][:4]
top4_feats = [feat_names[i] for i in top4_idx]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("SHAP Dependence Plots — Top 4 Features", fontsize=13, fontweight="bold")
for ax, feat in zip(axes.flatten(), top4_feats):
    shap.dependence_plot(feat, shap_values.values, shap_sample,
                         ax=ax, show=False, dot_size=8, alpha=0.4)
    ax.set_title(feat, fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "16_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/16_shap_dependence.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Waterfall: highest-risk at-risk student
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 4: SHAP waterfall — highest risk student …")
at_risk_mask = y_test_sample.values == 0
if at_risk_mask.sum() > 0:
    risk_probs     = probs_sample[at_risk_mask]
    top_local_idx  = np.argmin(risk_probs)   # lowest pass prob = highest risk
    global_idx     = np.where(at_risk_mask)[0][top_local_idx]

    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(shap_values[global_idx], max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Highest-Risk Student  (pass prob={probs_sample[global_idx]:.3f})",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "17_shap_waterfall_atrisk.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ outputs/17_shap_waterfall_atrisk.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Waterfall: borderline passing student
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 5: SHAP waterfall — borderline passing …")
border_mask = (y_test_sample.values == 1) & (probs_sample < 0.65)
if border_mask.sum() > 0:
    border_probs   = probs_sample[border_mask]
    border_local   = np.argmin(border_probs)
    border_global  = np.where(border_mask)[0][border_local]

    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(shap_values[border_global], max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Borderline Passing Student  (pass prob={probs_sample[border_global]:.3f})",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "18_shap_waterfall_borderline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ outputs/18_shap_waterfall_borderline.png")

# ── Top features ──────────────────────────────────────────────────────────────
shap_df = pd.DataFrame({"feature": feat_names, "mean_shap": mean_shap})
shap_df = shap_df.sort_values("mean_shap", ascending=False)

print("\n" + "="*55)
print("📋 Top 10 Features by Mean |SHAP|")
print("="*55)
print(shap_df.head(10).to_string(index=False))

print("\n" + "="*55)
print("✅ SHAP explainability complete!")
print("   Charts saved → outputs/ (14–18)")
print("   Next → python serving/api.py  (Phase 4)")
print("="*55)