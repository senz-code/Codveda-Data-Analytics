"""
=============================================================
Codveda Technology - Data Analytics Internship
Level 2 | Task 3: Clustering Analysis (K-Means)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid")
COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv("../../datasets/iris.csv")
X = df.drop("species", axis=1)
y_true = df["species"]

print("=" * 55)
print("LEVEL 2 – TASK 3: CLUSTERING ANALYSIS (K-MEANS)")
print("=" * 55)
print(f"\nShape: {df.shape}")

# ── 2. Standardise Features ──────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n[2] Features standardised with StandardScaler")

# ── 3. Elbow Method ──────────────────────────────────────────
inertia     = []
silhouettes = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Optimal K Selection", fontsize=14, fontweight="bold")
axes[0].plot(list(K_range), inertia, "bo-", linewidth=2, markersize=7)
axes[0].set_title("Elbow Method – Inertia")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].axvline(3, color="red", linestyle="--", label="K=3 (elbow)")
axes[0].legend()
axes[0].grid(alpha=0.4)

axes[1].plot(list(K_range), silhouettes, "rs-", linewidth=2, markersize=7)
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].axvline(3, color="blue", linestyle="--", label="K=3")
axes[1].legend()
axes[1].grid(alpha=0.4)
plt.tight_layout()
plt.savefig("plots/elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.close()
print("[3] Elbow & silhouette plots saved → plots/elbow_silhouette.png")

# ── 4. Fit K-Means with K=3 ──────────────────────────────────
optimal_k = 3
km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)
df["cluster"] = labels

ari = adjusted_rand_score(y_true, labels)
sil = silhouette_score(X_scaled, labels)
print(f"\n[4] K-Means (K={optimal_k}) fitted")
print(f"   Adjusted Rand Index : {ari:.4f}")
print(f"   Silhouette Score    : {sil:.4f}")

# Cluster sizes
print("\n   Cluster distribution:")
print(pd.Series(labels).value_counts().sort_index().to_string())

# ── 5. 2D Scatter – Original Features ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("K-Means Clustering – Iris Dataset", fontsize=14, fontweight="bold")

# K-Means clusters
for c in range(optimal_k):
    mask = labels == c
    axes[0].scatter(X.loc[mask,"petal_length"], X.loc[mask,"petal_width"],
                    color=COLORS[c], s=70, alpha=0.8, edgecolors="white",
                    label=f"Cluster {c}")
# Centroids (inverse-transform to original scale)
centroids_orig = scaler.inverse_transform(km.cluster_centers_)
axes[0].scatter(centroids_orig[:,2], centroids_orig[:,3],
                color="black", marker="X", s=200, zorder=5, label="Centroids")
axes[0].set_xlabel("Petal Length (cm)")
axes[0].set_ylabel("Petal Width (cm)")
axes[0].set_title("K-Means Clusters (K=3)")
axes[0].legend()
axes[0].grid(alpha=0.4)

# True labels
palette = {"setosa":"#e74c3c", "versicolor":"#2ecc71", "virginica":"#3498db"}
for sp, color in palette.items():
    mask = y_true == sp
    axes[1].scatter(X.loc[mask,"petal_length"], X.loc[mask,"petal_width"],
                    color=color, s=70, alpha=0.8, edgecolors="white",
                    label=sp.capitalize())
axes[1].set_xlabel("Petal Length (cm)")
axes[1].set_ylabel("Petal Width (cm)")
axes[1].set_title("True Species Labels")
axes[1].legend()
axes[1].grid(alpha=0.4)

plt.tight_layout()
plt.savefig("plots/cluster_scatter_2d.png", dpi=150, bbox_inches="tight")
plt.close()
print("[5] 2D cluster scatter saved → plots/cluster_scatter_2d.png")

# ── 6. PCA Visualisation ─────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_ratio = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(9, 6))
for c in range(optimal_k):
    mask = labels == c
    ax.scatter(X_pca[mask,0], X_pca[mask,1],
               color=COLORS[c], s=70, alpha=0.8, edgecolors="white",
               label=f"Cluster {c}")
centroids_pca = pca.transform(km.cluster_centers_)
ax.scatter(centroids_pca[:,0], centroids_pca[:,1],
           color="black", marker="X", s=250, zorder=5, label="Centroids")
ax.set_title(f"K-Means Clusters in PCA Space\n"
             f"(PC1={var_ratio[0]:.1%}, PC2={var_ratio[1]:.1%} variance)", fontsize=13)
ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} variance)")
ax.legend()
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("plots/cluster_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print("[6] PCA cluster plot saved → plots/cluster_pca.png")

print("\n✅  Task 3 complete!\n")
