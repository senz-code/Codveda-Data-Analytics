"""
=============================================================
Codveda Technology - Data Analytics Internship
Level 3 | Task 1: Predictive Modeling (Classification)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.model_selection import GridSearchCV
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid")

# ── 1. Load Data ─────────────────────────────────────────────
train = pd.read_csv("../../datasets/churn_train.csv")
test  = pd.read_csv("../../datasets/churn_test.csv")
df    = pd.concat([train, test], ignore_index=True)

print("=" * 55)
print("LEVEL 3 – TASK 1: PREDICTIVE MODELING (CLASSIFICATION)")
print("=" * 55)
print(f"\nTrain: {train.shape}  |  Test: {test.shape}")
print(f"Churn rate: {df['Churn'].mean():.2%}")

# ── 2. Preprocessing ─────────────────────────────────────────
# Encode binary/ordinal columns
le = LabelEncoder()
for col in ["International plan", "Voice mail plan"]:
    df[col] = le.fit_transform(df[col])

# Drop non-informative columns
df.drop(columns=["State", "Area code"], inplace=True)

# Target
df["Churn"] = df["Churn"].map({True: 1, False: 0,
                                "True": 1, "False": 0}).astype(int)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Recreate train/test with same proportions
X_train, y_train = X.iloc[:len(train)], y.iloc[:len(train)]
X_test,  y_test  = X.iloc[len(train):], y.iloc[len(train):]

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"\n[2] Preprocessed: {X_train.shape[1]} features")

# ── 3. Train Multiple Models ─────────────────────────────────
models = {
    "Decision Tree"      : DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
}
results = {}
for name, m in models.items():
    m.fit(X_train_s, y_train)
    pred = m.predict(X_test_s)
    results[name] = {
        "model"    : m,
        "preds"    : pred,
        "accuracy" : accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall"   : recall_score(y_test, pred),
        "f1"       : f1_score(y_test, pred),
    }
    print(f"\n[3] {name}")
    print(f"   Accuracy={results[name]['accuracy']:.4f}  "
          f"Precision={results[name]['precision']:.4f}  "
          f"Recall={results[name]['recall']:.4f}  "
          f"F1={results[name]['f1']:.4f}")

# ── 4. Metrics Comparison Plot ───────────────────────────────
metrics_df = pd.DataFrame({k: {m: v for m, v in r.items()
                                if m not in ("model","preds")}
                            for k, r in results.items()}).T
fig, ax = plt.subplots(figsize=(10, 5))
metrics_df.plot(kind="bar", ax=ax, edgecolor="white", width=0.7,
                color=["#3498db","#2ecc71","#e74c3c","#9b59b6"])
ax.set_title("Model Comparison – Evaluation Metrics", fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_xticklabels(metrics_df.index, rotation=15, ha="right")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[4] Model comparison saved → plots/model_comparison.png")

# ── 5. Confusion Matrices ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["preds"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn","Churn"], yticklabels=["No Churn","Churn"])
    ax.set_title(f"{name}\nF1={r['f1']:.3f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("[5] Confusion matrices saved → plots/confusion_matrices.png")

# ── 6. ROC Curves ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = ["#3498db","#e74c3c","#2ecc71"]
for (name, r), color in zip(results.items(), colors_roc):
    if hasattr(r["model"], "predict_proba"):
        proba = r["model"].predict_proba(X_test_s)[:,1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={roc_auc:.3f})")
ax.plot([0,1],[0,1],"k--", linewidth=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("plots/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("[6] ROC curves saved → plots/roc_curves.png")

# ── 7. Feature Importance (Random Forest) ────────────────────
rf = results["Random Forest"]["model"]
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
fi.head(15).plot(kind="barh", ax=ax, color="#3498db", edgecolor="white")
ax.invert_yaxis()
ax.set_title("Top 15 Feature Importances – Random Forest", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance")
ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("[7] Feature importance saved → plots/feature_importance.png")

# ── 8. Hyperparameter Tuning (Random Forest) ─────────────────
print("\n[8] Running GridSearchCV on Random Forest (this may take ~30s)...")
param_grid = {
    "n_estimators"    : [50, 100, 200],
    "max_depth"       : [None, 5, 10],
    "min_samples_split": [2, 5],
}
gs = GridSearchCV(RandomForestClassifier(random_state=42),
                  param_grid, cv=5, scoring="f1", n_jobs=-1)
gs.fit(X_train_s, y_train)
best = gs.best_estimator_
best_pred = best.predict(X_test_s)
print(f"   Best params : {gs.best_params_}")
print(f"   Best F1 (CV): {gs.best_score_:.4f}")
print(f"   Test F1     : {f1_score(y_test, best_pred):.4f}")
print(f"   Test Accuracy: {accuracy_score(y_test, best_pred):.4f}")

print("\n✅  Task 1 complete!\n")
