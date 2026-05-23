"""
=============================================================
Codveda Technology - Data Analytics Internship
Level 1 | Task 1: Data Cleaning and Preprocessing
Dataset : Iris CSV
=============================================================
"""

import pandas as pd
import numpy as np

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv("../../datasets/iris.csv")
print("=" * 55)
print("LEVEL 1 – TASK 1: DATA CLEANING & PREPROCESSING")
print("=" * 55)
print(f"\n[1] Dataset loaded  →  {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# ── 2. Basic Info ────────────────────────────────────────────
print("\n[2] Data types & non-null counts:")
print(df.info())

# ── 3. Missing Values ────────────────────────────────────────
print("\n[3] Missing values per column:")
print(df.isnull().sum())

# Artificially inject 5 missing values for demonstration
np.random.seed(42)
for col in ["sepal_length", "sepal_width", "petal_length"]:
    idx = np.random.choice(df.index, 2, replace=False)
    df.loc[idx, col] = np.nan

print("\n   After injecting missing values:")
print(df.isnull().sum())

# Impute numeric columns with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"   '{col}' → filled with median ({median_val:.3f})")

print("\n   Missing values after imputation:")
print(df.isnull().sum())

# ── 4. Duplicate Rows ────────────────────────────────────────
print(f"\n[4] Duplicate rows found: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"   Shape after removing duplicates: {df.shape}")

# ── 5. Standardise Categorical Variables ─────────────────────
print(f"\n[5] Unique species (before): {df['species'].unique()}")
# Inject inconsistent casing for demo
df.loc[0, "species"] = "SETOSA"
df.loc[1, "species"] = "  versicolor "
df["species"] = df["species"].str.strip().str.lower()
print(f"   Unique species (after):  {df['species'].unique()}")

# ── 6. Feature Engineering ───────────────────────────────────
df["sepal_ratio"] = (df["sepal_length"] / df["sepal_width"]).round(3)
df["petal_area"]  = (df["petal_length"] * df["petal_width"]).round(3)
print(f"\n[6] New features added: 'sepal_ratio', 'petal_area'")

# ── 7. Summary Statistics ────────────────────────────────────
print("\n[7] Summary statistics after cleaning:")
print(df.describe().round(3))

# ── 8. Save Cleaned Dataset ──────────────────────────────────
df.to_csv("iris_cleaned.csv", index=False)
print("\n[8] Cleaned dataset saved → iris_cleaned.csv")
print("\n✅  Task 1 complete!\n")
