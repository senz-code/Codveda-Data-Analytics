"""
=============================================================
Codveda Technology - Data Analytics Internship
Level 3 | Task 2: Building an Analytics Dashboard
Dataset : Churn + Stock Prices + Iris
Note    : Static multi-panel dashboard (matplotlib)
          For a live interactive version, open task2_dashboard.html
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid")
PALETTE = ["#3498db","#e74c3c","#2ecc71","#f39c12","#9b59b6","#1abc9c"]

# ── Load Datasets ────────────────────────────────────────────
churn = pd.concat([
    pd.read_csv("../../datasets/churn_train.csv"),
    pd.read_csv("../../datasets/churn_test.csv")
], ignore_index=True)
churn["Churn"] = churn["Churn"].astype(str).str.strip()

stocks = pd.read_csv("../../datasets/stock_prices.csv")
stocks.columns = stocks.columns.str.strip().str.lower()
stocks["date"] = pd.to_datetime(stocks["date"])

iris = pd.read_csv("../../datasets/iris.csv")

print("=" * 55)
print("LEVEL 3 – TASK 2: ANALYTICS DASHBOARD")
print("=" * 55)

# ═══════════════════════════════════════════════════════════
#  DASHBOARD 1 – CHURN ANALYSIS
# ═══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.suptitle("📊 Customer Churn Analytics Dashboard", fontsize=20,
             fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1a – Churn rate overall (pie)
ax1 = fig.add_subplot(gs[0, 0])
churn_counts = churn["Churn"].value_counts()
ax1.pie(churn_counts.values, labels=churn_counts.index,
        colors=[PALETTE[1], PALETTE[0]], autopct="%1.1f%%",
        startangle=140, wedgeprops={"edgecolor":"white","linewidth":2})
ax1.set_title("Overall Churn Rate", fontsize=12, fontweight="bold")

# 1b – Churn by International Plan (bar)
ax2 = fig.add_subplot(gs[0, 1])
ip_churn = churn.groupby("International plan")["Churn"].apply(
    lambda x: (x.astype(str) == "True").mean() * 100).reset_index()
bars = ax2.bar(ip_churn["International plan"], ip_churn["Churn"],
               color=[PALETTE[0], PALETTE[1]], edgecolor="white", width=0.5)
for bar, val in zip(bars, ip_churn["Churn"]):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val:.1f}%", ha="center", fontweight="bold", fontsize=10)
ax2.set_title("Churn Rate by Int'l Plan", fontsize=12, fontweight="bold")
ax2.set_ylabel("Churn Rate (%)"); ax2.grid(axis="y", alpha=0.4)

# 1c – Customer service calls vs churn (box)
ax3 = fig.add_subplot(gs[0, 2])
churn_bool = churn["Churn"].astype(str) == "True"
no_churn   = churn.loc[~churn_bool, "Customer service calls"]
yes_churn  = churn.loc[churn_bool,  "Customer service calls"]
bp = ax3.boxplot([no_churn, yes_churn], labels=["No Churn","Churned"],
                 patch_artist=True, widths=0.5,
                 medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], [PALETTE[0], PALETTE[1]]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax3.set_title("Service Calls Distribution", fontsize=12, fontweight="bold")
ax3.set_ylabel("Customer Service Calls"); ax3.grid(axis="y", alpha=0.4)

# 1d – Total day minutes vs churn (histogram)
ax4 = fig.add_subplot(gs[1, 0:2])
ax4.hist(no_churn.index.map(churn["Total day minutes"]), bins=40,
         alpha=0.6, color=PALETTE[0], label="No Churn", edgecolor="white")
ax4.hist(yes_churn.index.map(churn["Total day minutes"]), bins=40,
         alpha=0.6, color=PALETTE[1], label="Churned", edgecolor="white")
ax4.set_title("Total Day Minutes Distribution by Churn", fontsize=12, fontweight="bold")
ax4.set_xlabel("Total Day Minutes"); ax4.set_ylabel("Frequency")
ax4.legend(); ax4.grid(alpha=0.4)

# 1e – Churn by state (top 10)
ax5 = fig.add_subplot(gs[1, 2])
state_churn = churn.groupby("State")["Churn"].apply(
    lambda x: (x.astype(str)=="True").mean()*100).nlargest(10)
ax5.barh(state_churn.index[::-1], state_churn.values[::-1],
         color=PALETTE[4], edgecolor="white")
ax5.set_title("Top 10 States by Churn Rate", fontsize=12, fontweight="bold")
ax5.set_xlabel("Churn Rate (%)"); ax5.grid(axis="x", alpha=0.4)

# 1f – Churn rate by number of VM messages
ax6 = fig.add_subplot(gs[2, :])
vm_bins = pd.cut(churn["Number vmail messages"], bins=10)
vm_churn_rate = churn.groupby(vm_bins, observed=True)["Churn"].apply(
    lambda x: (x.astype(str)=="True").mean()*100)
ax6.bar(range(len(vm_churn_rate)), vm_churn_rate.values,
        color=PALETTE[2], edgecolor="white", alpha=0.8)
ax6.set_xticks(range(len(vm_churn_rate)))
ax6.set_xticklabels([str(b) for b in vm_churn_rate.index], rotation=45, ha="right", fontsize=8)
ax6.set_title("Churn Rate by Number of Voicemail Messages", fontsize=12, fontweight="bold")
ax6.set_xlabel("Voicemail Message Bins"); ax6.set_ylabel("Churn Rate (%)")
ax6.grid(axis="y", alpha=0.4)

plt.savefig("plots/dashboard_churn.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Dashboard 1] Churn dashboard saved → plots/dashboard_churn.png")

# ═══════════════════════════════════════════════════════════
#  DASHBOARD 2 – STOCK MARKET ANALYSIS
# ═══════════════════════════════════════════════════════════
aapl = stocks[stocks["symbol"]=="AAPL"].sort_values("date").set_index("date")
top5_vol = stocks.groupby("symbol")["volume"].sum().nlargest(5)

fig2, axes2 = plt.subplots(2, 2, figsize=(18, 10))
fig2.suptitle("📈 Stock Market Analytics Dashboard", fontsize=18,
              fontweight="bold")

# 2a – AAPL price with MA
axes2[0,0].plot(aapl.index, aapl["close"], color="#bdc3c7", linewidth=0.8,
                alpha=0.7, label="Daily Close")
axes2[0,0].plot(aapl.index, aapl["close"].rolling(30).mean(),
                color=PALETTE[1], linewidth=2, label="30-day MA")
axes2[0,0].plot(aapl.index, aapl["close"].rolling(90).mean(),
                color=PALETTE[4], linewidth=2, label="90-day MA")
axes2[0,0].set_title("AAPL Close Price + Moving Averages", fontsize=12, fontweight="bold")
axes2[0,0].set_ylabel("Price (USD)"); axes2[0,0].legend()
axes2[0,0].grid(alpha=0.4); plt.setp(axes2[0,0].xaxis.get_majorticklabels(), rotation=30)

# 2b – Top 5 stocks by total volume
axes2[0,1].bar(top5_vol.index, top5_vol.values/1e9, color=PALETTE[:5], edgecolor="white")
axes2[0,1].set_title("Top 5 Stocks by Total Volume", fontsize=12, fontweight="bold")
axes2[0,1].set_ylabel("Volume (Billions)"); axes2[0,1].grid(axis="y", alpha=0.4)

# 2c – Daily returns histogram
returns = aapl["close"].pct_change().dropna() * 100
axes2[1,0].hist(returns, bins=50, color=PALETTE[0], edgecolor="white", alpha=0.8)
axes2[1,0].axvline(returns.mean(), color=PALETTE[1], linewidth=2,
                    linestyle="--", label=f"Mean={returns.mean():.2f}%")
axes2[1,0].set_title("AAPL Daily Returns Distribution", fontsize=12, fontweight="bold")
axes2[1,0].set_xlabel("Daily Return (%)"); axes2[1,0].set_ylabel("Frequency")
axes2[1,0].legend(); axes2[1,0].grid(alpha=0.4)

# 2d – Normalised multi-stock comparison
for sym, color in zip(["AAPL","GOOG","MSFT"], PALETTE):
    if sym in stocks["symbol"].values:
        sub = stocks[stocks["symbol"]==sym].sort_values("date").set_index("date")["close"]
        axes2[1,1].plot(sub.index, sub/sub.iloc[0]*100, linewidth=1.5,
                        label=sym, color=color)
axes2[1,1].set_title("Normalised Stock Performance (Base=100)", fontsize=12, fontweight="bold")
axes2[1,1].set_ylabel("Normalised Price"); axes2[1,1].legend()
axes2[1,1].grid(alpha=0.4); plt.setp(axes2[1,1].xaxis.get_majorticklabels(), rotation=30)

plt.tight_layout()
plt.savefig("plots/dashboard_stocks.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Dashboard 2] Stock dashboard saved → plots/dashboard_stocks.png")

# ═══════════════════════════════════════════════════════════
#  DASHBOARD 3 – IRIS EDA
# ═══════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
fig3.suptitle("🌸 Iris Dataset Analytics Dashboard", fontsize=18, fontweight="bold")
species_palette = {"setosa":PALETTE[1], "versicolor":PALETTE[2], "virginica":PALETTE[0]}

# 3a – Petal scatter
for sp, color in species_palette.items():
    sub = iris[iris["species"]==sp]
    axes3[0,0].scatter(sub["petal_length"], sub["petal_width"],
                       color=color, s=70, alpha=0.8, label=sp.capitalize(), edgecolors="white")
axes3[0,0].set_title("Petal Length vs Width", fontsize=12, fontweight="bold")
axes3[0,0].set_xlabel("Petal Length (cm)"); axes3[0,0].set_ylabel("Petal Width (cm)")
axes3[0,0].legend(); axes3[0,0].grid(alpha=0.4)

# 3b – Sepal scatter
for sp, color in species_palette.items():
    sub = iris[iris["species"]==sp]
    axes3[0,1].scatter(sub["sepal_length"], sub["sepal_width"],
                       color=color, s=70, alpha=0.8, label=sp.capitalize(), edgecolors="white")
axes3[0,1].set_title("Sepal Length vs Width", fontsize=12, fontweight="bold")
axes3[0,1].set_xlabel("Sepal Length (cm)"); axes3[0,1].set_ylabel("Sepal Width (cm)")
axes3[0,1].legend(); axes3[0,1].grid(alpha=0.4)

# 3c – Species pie
sp_counts = iris["species"].value_counts()
axes3[0,2].pie(sp_counts, labels=sp_counts.index.str.capitalize(),
               colors=list(species_palette.values()), autopct="%1.1f%%",
               startangle=140, wedgeprops={"edgecolor":"white"})
axes3[0,2].set_title("Species Distribution", fontsize=12, fontweight="bold")

# 3d-3f – Feature boxplots
for ax, feat in zip(axes3[1], ["sepal_length","petal_length","petal_width"]):
    data_by_species = [iris[iris["species"]==sp][feat] for sp in species_palette]
    bp = ax.boxplot(data_by_species, labels=[s.capitalize() for s in species_palette],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], species_palette.values()):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title(f"{feat.replace('_',' ').title()} by Species", fontsize=12, fontweight="bold")
    ax.set_ylabel("cm"); ax.grid(axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig("plots/dashboard_iris.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Dashboard 3] Iris dashboard saved → plots/dashboard_iris.png")
print("\n✅  Task 2 complete!\n")
