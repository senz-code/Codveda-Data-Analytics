"""
=============================================================
Codveda Technology - Data Analytics Internship
Level 1 | Task 3: Basic Data Visualization
Dataset : Iris CSV
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
sns.set_theme(style="whitegrid")
COLORS = ["#2ecc71", "#3498db", "#e74c3c"]

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv("../../datasets/iris.csv")
print("=" * 55)
print("LEVEL 1 – TASK 3: BASIC DATA VISUALIZATION")
print("=" * 55)

# ── 2. Bar Plot – Mean feature values per species ────────────
means = df.groupby("species")[["sepal_length","sepal_width",
                                "petal_length","petal_width"]].mean()
fig, ax = plt.subplots(figsize=(10, 6))
means.T.plot(kind="bar", ax=ax, color=COLORS, edgecolor="white", width=0.7)
ax.set_title("Mean Feature Values per Species", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Feature", fontsize=12)
ax.set_ylabel("Mean Value (cm)", fontsize=12)
ax.set_xticklabels([l.replace("_", "\n") for l in means.columns], rotation=0)
ax.legend(title="Species", fontsize=10)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("plots/bar_plot_species_means.png", dpi=150, bbox_inches="tight")
plt.close()
print("[2] Bar plot saved → plots/bar_plot_species_means.png")

# ── 3. Line Chart – Sorted sepal length trend ────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for sp, grp in df.groupby("species"):
    sorted_vals = grp["sepal_length"].reset_index(drop=True).sort_values().values
    ax.plot(sorted_vals, label=sp.capitalize(), linewidth=2)
ax.set_title("Sepal Length Trend by Species (sorted)", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Sample Index (sorted)", fontsize=12)
ax.set_ylabel("Sepal Length (cm)", fontsize=12)
ax.legend(title="Species")
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("plots/line_chart_sepal_length.png", dpi=150, bbox_inches="tight")
plt.close()
print("[3] Line chart saved → plots/line_chart_sepal_length.png")

# ── 4. Scatter Plot – Petal dims coloured by species ─────────
fig, ax = plt.subplots(figsize=(9, 6))
for sp, color in zip(df["species"].unique(), COLORS):
    sub = df[df["species"] == sp]
    ax.scatter(sub["petal_length"], sub["petal_width"],
               label=sp.capitalize(), color=color, s=70, alpha=0.8, edgecolors="white")
ax.set_title("Petal Length vs Petal Width", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Petal Length (cm)", fontsize=12)
ax.set_ylabel("Petal Width (cm)", fontsize=12)
ax.legend(title="Species")
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("plots/scatter_petal_dims.png", dpi=150, bbox_inches="tight")
plt.close()
print("[4] Scatter plot saved → plots/scatter_petal_dims.png")

# ── 5. Histogram – Petal length distribution ─────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for sp, color in zip(df["species"].unique(), COLORS):
    sub = df[df["species"] == sp]
    ax.hist(sub["petal_length"], bins=15, alpha=0.6, label=sp.capitalize(),
            color=color, edgecolor="white")
ax.set_title("Petal Length Distribution by Species", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Petal Length (cm)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.legend(title="Species")
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("plots/histogram_petal_length.png", dpi=150, bbox_inches="tight")
plt.close()
print("[5] Histogram saved → plots/histogram_petal_length.png")

# ── 6. Combined Dashboard ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Iris Dataset – Visual Summary Dashboard", fontsize=17, fontweight="bold")

# 6a Bar
means.T.plot(kind="bar", ax=axes[0,0], color=COLORS, edgecolor="white", width=0.7, legend=False)
axes[0,0].set_title("Mean Feature Values per Species")
axes[0,0].set_xticklabels([l.replace("_","\n") for l in means.columns], rotation=0, fontsize=8)
axes[0,0].grid(axis="y", alpha=0.4)

# 6b Scatter
for sp, color in zip(df["species"].unique(), COLORS):
    sub = df[df["species"] == sp]
    axes[0,1].scatter(sub["sepal_length"], sub["sepal_width"],
                      label=sp.capitalize(), color=color, s=55, alpha=0.8)
axes[0,1].set_title("Sepal Length vs Sepal Width")
axes[0,1].legend(fontsize=8)
axes[0,1].grid(alpha=0.4)

# 6c Boxplot
df_melt = df.melt(id_vars="species", value_vars=["sepal_length","petal_length"],
                  var_name="feature", value_name="cm")
sns.boxplot(data=df_melt, x="feature", y="cm", hue="species",
            palette=COLORS, ax=axes[1,0])
axes[1,0].set_title("Sepal vs Petal Length Boxplot")
axes[1,0].legend(fontsize=8)

# 6d Pie
counts = df["species"].value_counts()
axes[1,1].pie(counts, labels=counts.index.str.capitalize(),
              colors=COLORS, autopct="%1.1f%%", startangle=140,
              wedgeprops={"edgecolor":"white"})
axes[1,1].set_title("Species Proportion")

plt.tight_layout()
plt.savefig("plots/dashboard_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("[6] Dashboard saved → plots/dashboard_summary.png")

print("\n✅  Task 3 complete!\n")
