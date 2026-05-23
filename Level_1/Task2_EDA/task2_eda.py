"""
=============================================================
Codveda Technology - Data Analytics Internship
Level 1 | Task 2: Exploratory Data Analysis (EDA)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="Set2")

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv("../../datasets/iris.csv")
print("=" * 55)
print("LEVEL 1 – TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 55)
print(f"\nShape: {df.shape}")
print(df.head())

# ── 2. Summary Statistics ────────────────────────────────────
print("\n[2] Summary Statistics:")
print(df.describe().round(3))

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Mean, Median, Mode, Std per numeric column
print("\n[2b] Mean | Median | Mode | Std per feature:")
for col in numeric_cols:
    m  = df[col].mean()
    md = df[col].median()
    mo = df[col].mode()[0]
    s  = df[col].std()
    print(f"   {col:<15} mean={m:.3f}  median={md:.3f}  mode={mo:.3f}  std={s:.3f}")

# ── 3. Class Distribution ────────────────────────────────────
print("\n[3] Class distribution:")
print(df["species"].value_counts())

# ── 4. Histograms ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Iris Feature Distributions", fontsize=16, fontweight="bold")
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.histplot(data=df, x=col, hue="species", kde=True, ax=ax, bins=20)
    ax.set_title(col.replace("_", " ").title())
plt.tight_layout()
plt.savefig("plots/histograms.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[4] Histograms saved → plots/histograms.png")

# ── 5. Boxplots ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Iris Feature Boxplots by Species", fontsize=16, fontweight="bold")
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.boxplot(data=df, x="species", y=col, ax=ax, palette="Set2")
    ax.set_title(col.replace("_", " ").title())
plt.tight_layout()
plt.savefig("plots/boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("[5] Boxplots saved → plots/boxplots.png")

# ── 6. Scatter Plots ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Iris Scatter Plots", fontsize=16, fontweight="bold")
sns.scatterplot(data=df, x="sepal_length", y="sepal_width",  hue="species", ax=axes[0], s=80)
axes[0].set_title("Sepal Length vs Sepal Width")
sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species", ax=axes[1], s=80)
axes[1].set_title("Petal Length vs Petal Width")
plt.tight_layout()
plt.savefig("plots/scatter_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("[6] Scatter plots saved → plots/scatter_plots.png")

# ── 7. Correlation Heatmap ───────────────────────────────────
corr = df[numeric_cols].corr()
print("\n[7] Correlation Matrix:")
print(corr.round(3))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
            linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap – Iris Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("[7] Correlation heatmap saved → plots/correlation_heatmap.png")

# ── 8. Pair Plot ─────────────────────────────────────────────
pairplot = sns.pairplot(df, hue="species", diag_kind="kde", plot_kws={"alpha": 0.6})
pairplot.fig.suptitle("Iris Pair Plot", y=1.02, fontsize=14, fontweight="bold")
pairplot.savefig("plots/pair_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("[8] Pair plot saved → plots/pair_plot.png")

print("\n✅  Task 2 complete!\n")
