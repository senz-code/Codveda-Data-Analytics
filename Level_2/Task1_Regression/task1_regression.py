"""
=============================================================
Codveda Technology - Data Analytics Internship
Level 2 | Task 1: Regression Analysis
Dataset : House Prices CSV (Boston-style)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid")

# ── 1. Load Dataset ──────────────────────────────────────────
# Boston-style dataset: last column is MEDV (house price)
cols = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE",
        "DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
df = pd.read_csv("../../datasets/house_prices.csv",
                 header=None, names=cols, sep=r"\s+", engine="python")

print("=" * 55)
print("LEVEL 2 – TASK 1: REGRESSION ANALYSIS")
print("=" * 55)
print(f"\nShape: {df.shape}")
print(df.head())
print("\nMissing values:", df.isnull().sum().sum())

# ── 2. Feature / Target Split ────────────────────────────────
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# ── 3. Train / Test Split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"\n[3] Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ── 4. Feature Scaling ───────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 5. Fit Linear Regression ─────────────────────────────────
model = LinearRegression()
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

# ── 6. Model Evaluation ──────────────────────────────────────
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n[6] Model Evaluation Metrics:")
print(f"   R²   : {r2:.4f}")
print(f"   MSE  : {mse:.4f}")
print(f"   RMSE : {rmse:.4f}")
print(f"   MAE  : {mae:.4f}")

# ── 7. Coefficients ──────────────────────────────────────────
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print("\n[7] Feature Coefficients (sorted by importance):")
print(coef_df.to_string(index=False))

# ── 8. Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Regression Analysis – House Prices", fontsize=15, fontweight="bold")

# Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.6, color="#3498db", edgecolors="white", s=60)
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect Fit")
axes[0].set_xlabel("Actual Price ($000s)")
axes[0].set_ylabel("Predicted Price ($000s)")
axes[0].set_title(f"Actual vs Predicted\nR² = {r2:.4f}")
axes[0].legend()

# Residuals
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6, color="#e74c3c", edgecolors="white", s=60)
axes[1].axhline(0, color="black", linewidth=1.5, linestyle="--")
axes[1].set_xlabel("Predicted Price ($000s)")
axes[1].set_ylabel("Residual")
axes[1].set_title("Residual Plot")

# Coefficient Bar Chart
colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coef_df["Coefficient"]]
axes[2].barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="white")
axes[2].axvline(0, color="black", linewidth=1)
axes[2].set_title("Feature Coefficients")
axes[2].set_xlabel("Coefficient Value")

plt.tight_layout()
plt.savefig("plots/regression_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[8] Plots saved → plots/regression_analysis.png")
print("\n✅  Task 1 complete!\n")
