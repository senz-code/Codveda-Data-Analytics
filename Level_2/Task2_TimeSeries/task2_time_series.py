"""
=============================================================
Codveda Technology - Data Analytics Internship
Level 2 | Task 2: Time Series Analysis
Dataset : Stock Prices CSV
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("plots", exist_ok=True)

# ── 1. Load & Prepare Data ───────────────────────────────────
df = pd.read_csv("../../datasets/stock_prices.csv")
df.columns = df.columns.str.strip().str.lower()
df["date"] = pd.to_datetime(df["date"])

print("=" * 55)
print("LEVEL 2 – TASK 2: TIME SERIES ANALYSIS")
print("=" * 55)
print(f"\nDataset shape: {df.shape}")
print(f"Symbols: {list(df['symbol'].unique()[:8])} ...")
print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

symbol = "AAPL"
ts = df[df["symbol"] == symbol].sort_values("date").set_index("date")["close"]
print(f"\n[1] Analysing '{symbol}' — {len(ts)} trading days")

# ── 2. Raw Time Series Plot ───────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ts.index, ts.values, color="#3498db", linewidth=1.2)
ax.set_title(f"{symbol} – Daily Close Price", fontsize=14, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45); ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("plots/raw_time_series.png", dpi=150, bbox_inches="tight")
plt.close()
print("[2] Raw time series saved → plots/raw_time_series.png")

# ── 3. Moving Averages ───────────────────────────────────────
ts_df = ts.to_frame(name="close")
ts_df["MA_7"]  = ts_df["close"].rolling(7).mean()
ts_df["MA_30"] = ts_df["close"].rolling(30).mean()
ts_df["MA_90"] = ts_df["close"].rolling(90).mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ts_df.index, ts_df["close"],  color="#bdc3c7", linewidth=0.8, alpha=0.7, label="Daily Close")
ax.plot(ts_df.index, ts_df["MA_7"],   color="#e74c3c", linewidth=1.5, label="7-day MA")
ax.plot(ts_df.index, ts_df["MA_30"],  color="#2ecc71", linewidth=1.8, label="30-day MA")
ax.plot(ts_df.index, ts_df["MA_90"],  color="#9b59b6", linewidth=2.0, label="90-day MA")
ax.set_title(f"{symbol} – Moving Average Smoothing", fontsize=14, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45); ax.legend(); ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("plots/moving_averages.png", dpi=150, bbox_inches="tight")
plt.close()
print("[3] Moving averages saved → plots/moving_averages.png")

# ── 4. Manual Decomposition ───────────────────────────────────
# Trend: 30-day rolling mean
# Seasonality: deviation of weekly means from overall mean
# Residual: actual - trend - seasonality
ts_weekly = ts.resample("W").mean().dropna()
trend = ts_weekly.rolling(window=12, center=True).mean()
detrended = ts_weekly - trend
# Seasonal component: average weekly pattern (by week-of-year)
seasonal_map = detrended.groupby(detrended.index.isocalendar().week).mean()
seasonal = detrended.index.isocalendar().week.map(seasonal_map)
seasonal = pd.Series(seasonal.values, index=detrended.index)
residual = ts_weekly - trend - seasonal

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f"{symbol} – Time Series Decomposition (Weekly)", fontsize=14, fontweight="bold")
for ax, (data, title, color) in zip(axes, [
    (ts_weekly, "Observed",   "#3498db"),
    (trend,     "Trend",      "#e74c3c"),
    (seasonal,  "Seasonality","#2ecc71"),
    (residual,  "Residuals",  "#e67e22"),
]):
    ax.plot(data, color=color, linewidth=1.3)
    ax.set_ylabel(title, fontsize=11); ax.grid(alpha=0.4)
axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("plots/decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print("[4] Decomposition saved → plots/decomposition.png")

# ── 5. Daily Returns & Volatility ────────────────────────────
returns = ts.pct_change().dropna() * 100
rolling_vol = returns.rolling(30).std()

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle(f"{symbol} – Returns & 30-day Rolling Volatility", fontsize=14, fontweight="bold")
axes[0].plot(returns.index, returns.values, color="#3498db", linewidth=0.7, alpha=0.8)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_ylabel("Daily Return (%)"); axes[0].grid(alpha=0.4)
axes[1].plot(rolling_vol.index, rolling_vol.values, color="#e74c3c", linewidth=1.5)
axes[1].set_ylabel("30-day Rolling Volatility (%)"); axes[1].set_xlabel("Date")
axes[1].grid(alpha=0.4)
plt.tight_layout()
plt.savefig("plots/returns_volatility.png", dpi=150, bbox_inches="tight")
plt.close()
print("[5] Returns & volatility saved → plots/returns_volatility.png")

# ── 6. Multi-Symbol Comparison ───────────────────────────────
top_symbols = ["AAPL","GOOG","MSFT","AMZN"]
available   = [s for s in top_symbols if s in df["symbol"].values]
fig, ax = plt.subplots(figsize=(14, 6))
colors = ["#3498db","#e74c3c","#2ecc71","#f39c12"]
for sym, color in zip(available, colors):
    sub = df[df["symbol"] == sym].sort_values("date").set_index("date")["close"]
    normalised = sub / sub.iloc[0] * 100
    ax.plot(normalised.index, normalised.values, linewidth=1.5, label=sym, color=color)
ax.set_title("Normalised Price Performance (Base = 100)", fontsize=14, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Normalised Price")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45); ax.legend(title="Symbol"); ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("plots/multi_symbol_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[6] Multi-symbol comparison saved → plots/multi_symbol_comparison.png")
print("\n✅  Task 2 complete!\n")
