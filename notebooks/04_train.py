"""
notebooks/04_train.py
──────────────────────
Phase 3 → Model Training

What it does:
    - Loads processed train/test splits
    - Preprocesses features via sklearn pipeline
    - Tunes XGBoost with Optuna (50 trials)
    - Evaluates with precision, recall, F1, ROC-AUC
    - Finds optimal F1 threshold
    - Saves model + evaluation charts

Run:
    python notebooks/04_train.py
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score,
)
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
MODEL_DIR  = Path("models")
OUTPUT_DIR = Path("outputs")
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_TRIALS     = 50

PASS_COLOR = "#2ecc71"
FAIL_COLOR = "#e74c3c"
sns.set_theme(style="darkgrid")

# ── Load ──────────────────────────────────────────────────────────────────────
print("📂 Loading splits and preprocessor …")
X_train_raw = pd.read_parquet(DATA_DIR / "X_train.parquet")
X_test_raw  = pd.read_parquet(DATA_DIR / "X_test.parquet")
y_train     = pd.read_parquet(DATA_DIR / "y_train.parquet").squeeze()
y_test      = pd.read_parquet(DATA_DIR / "y_test.parquet").squeeze()
preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
FEATURES     = joblib.load(DATA_DIR  / "feature_list.joblib")

# ── Preprocess ────────────────────────────────────────────────────────────────
print("⚙️  Transforming features …")
X_train = preprocessor.transform(X_train_raw)
X_test  = preprocessor.transform(X_test_raw)

print(f"   Train : {X_train.shape}  | At-risk={(y_train==0).sum():,} ({(y_train==0).mean()*100:.1f}%)")
print(f"   Test  : {X_test.shape}   | At-risk={(y_test==0).sum():,}  ({(y_test==0).mean()*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n🔍 Optuna search — {N_TRIALS} trials …")

def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma":            trial.suggest_float("gamma", 0, 5),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "eval_metric":      "auc",
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    probs = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, probs)

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_params.update({
    "eval_metric":  "auc",
    "random_state": RANDOM_STATE,
    "n_jobs":       -1,
})

print(f"\n   Best ROC-AUC : {study.best_value:.4f}")
print(f"   Best params  : {best_params}")

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN FINAL MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🏋️  Training final model …")
model = XGBClassifier(**best_params)
model.fit(X_train, y_train, verbose=False)

y_prob = model.predict_proba(X_test)[:, 1]

# ── Optimal threshold ─────────────────────────────────────────────────────────
print("📐 Finding optimal threshold …")
thresholds = np.linspace(0.01, 0.99, 200)
f1_scores  = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]
best_f1     = max(f1_scores)
y_pred      = (y_prob >= best_thresh).astype(int)

# ── Metrics ───────────────────────────────────────────────────────────────────
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)
prec    = precision_score(y_test, y_pred, zero_division=0)
rec     = recall_score(y_test, y_pred, zero_division=0)
f1      = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "="*55)
print("📊 Evaluation Results")
print("="*55)
print(f"   ROC-AUC   : {roc_auc:.4f}")
print(f"   PR-AUC    : {pr_auc:.4f}")
print(f"   Precision : {prec:.4f}")
print(f"   Recall    : {rec:.4f}")
print(f"   F1 Score  : {f1:.4f}")
print(f"   Threshold : {best_thresh:.3f}")
print("\n" + classification_report(y_test, y_pred, target_names=["At-Risk", "Passed"]))

# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Generating evaluation charts …")

# ── Chart 1: ROC + PR ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Model Evaluation — XGBoost Student Performance", fontsize=13, fontweight="bold")

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, color=FAIL_COLOR, lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
axes[0].plot([0,1],[0,1], "k--", lw=1)
axes[0].fill_between(fpr, tpr, alpha=0.1, color=FAIL_COLOR)
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve"); axes[0].legend()

prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
axes[1].plot(rec_c, prec_c, color="#3498db", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
axes[1].axhline(y_test.mean(), linestyle="--", color="gray", lw=1,
                label=f"Baseline ({y_test.mean():.2f})")
axes[1].fill_between(rec_c, prec_c, alpha=0.1, color="#3498db")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve"); axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/08_roc_pr_curves.png")

# ── Chart 2: Confusion Matrix ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Confusion Matrix (threshold={best_thresh:.3f})", fontsize=13, fontweight="bold")
for ax, matrix, fmt, title in [
    (axes[0], confusion_matrix(y_test, y_pred),              "d",   "Counts"),
    (axes[1], confusion_matrix(y_test, y_pred, normalize="true"), ".2%", "Normalised"),
]:
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=["Pred At-Risk","Pred Passed"],
                yticklabels=["True At-Risk","True Passed"],
                ax=ax, linewidths=1)
    ax.set_title(title)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/09_confusion_matrix.png")

# ── Chart 3: Feature Importance ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 9))

# Get feature names after OHE
num_feats  = joblib.load(DATA_DIR / "numeric_features.joblib")
cat_feats  = joblib.load(DATA_DIR / "categorical_features.joblib")
ohe        = preprocessor.named_transformers_["cat"]["ohe"]
cat_names  = list(ohe.get_feature_names_out(cat_feats))
feat_names = num_feats + cat_names

importance = pd.Series(model.feature_importances_, index=feat_names).sort_values()
top20      = importance.tail(20)
colors_fi  = ["#e74c3c" if v > importance.median() else "#95a5a6" for v in top20]
top20.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="white")
ax.set_title("XGBoost Feature Importance (Top 20)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/10_feature_importance.png")

# ── Chart 4: Score Distribution ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y_prob[y_test == 1], bins=60, alpha=0.6, color=PASS_COLOR, label="Passed",    density=True)
ax.hist(y_prob[y_test == 0], bins=60, alpha=0.7, color=FAIL_COLOR, label="At-Risk",   density=True)
ax.axvline(best_thresh, color="#f0a500", linestyle="--", lw=2,
           label=f"Threshold = {best_thresh:.3f}")
ax.set_title("Predicted Probability Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Pass Probability"); ax.set_ylabel("Density"); ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/11_score_distribution.png")

# ── Chart 5: Optuna History ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Optuna Hyperparameter Search", fontsize=13, fontweight="bold")

trial_vals = [t.value for t in study.trials]
axes[0].plot(trial_vals, alpha=0.5, color="#3498db", lw=1, marker="o", markersize=3)
axes[0].plot(pd.Series(trial_vals).cummax(), color=FAIL_COLOR, lw=2, label="Best so far")
axes[0].set_title("ROC-AUC per Trial"); axes[0].set_xlabel("Trial")
axes[0].set_ylabel("ROC-AUC"); axes[0].legend()

try:
    importances = optuna.importance.get_param_importances(study)
    top_p = dict(list(importances.items())[:6])
    axes[1].barh(list(top_p.keys()), list(top_p.values()), color="#3498db", edgecolor="white")
    axes[1].set_title("Hyperparameter Importances"); axes[1].set_xlabel("Importance")
except Exception:
    axes[1].text(0.5, 0.5, "Not available", ha="center", transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "12_optuna_history.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/12_optuna_history.png")

# ── Chart 6: Threshold Analysis ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds, f1_scores, color="#3498db", lw=2)
ax.axvline(best_thresh, color=FAIL_COLOR, linestyle="--", lw=2,
           label=f"Best = {best_thresh:.3f}  (F1={best_f1:.4f})")
ax.set_title("F1 Score vs Decision Threshold", fontsize=13, fontweight="bold")
ax.set_xlabel("Threshold"); ax.set_ylabel("F1 Score"); ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "13_threshold_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/13_threshold_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE MODEL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n💾 Saving model artifacts …")
joblib.dump(model,       MODEL_DIR / "xgb_student_model.joblib")
joblib.dump(best_thresh, MODEL_DIR / "optimal_threshold.joblib")
joblib.dump(best_params, MODEL_DIR / "best_params.joblib")
joblib.dump({
    "roc_auc":   roc_auc,
    "pr_auc":    pr_auc,
    "precision": prec,
    "recall":    rec,
    "f1":        f1,
    "threshold": best_thresh,
}, MODEL_DIR / "eval_metrics.joblib")

print(f"   ✅ models/xgb_student_model.joblib")
print(f"   ✅ models/optimal_threshold.joblib")
print(f"   ✅ models/eval_metrics.joblib")

print("\n" + "="*55)
print(f"✅ Training complete!")
print(f"   ROC-AUC={roc_auc:.4f} | F1={f1:.4f} | Threshold={best_thresh:.3f}")
print("   Next → python notebooks/05_explain.py")
print("="*55)