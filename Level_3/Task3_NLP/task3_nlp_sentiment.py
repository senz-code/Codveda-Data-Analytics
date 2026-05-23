"""
=============================================================
Codveda Technology - Data Analytics Internship
Level 3 | Task 3: NLP – Sentiment Analysis
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import re, string, os, warnings
from collections import Counter
warnings.filterwarnings("ignore")

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid")

# ── Common English stop words (built-in, no NLTK needed) ────
STOP_WORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further get got had hadn't
has hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought
our ours ourselves out over own same shan't she she'd she'll she's should shouldn't
so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's will with won't would wouldn't you you'd
you'll you're you've your yours yourself yourselves
""".split())

# ── Simple rule-based sentiment lexicon ─────────────────────
POSITIVE_WORDS = set("""
good great excellent wonderful amazing fantastic beautiful happy joy love like best
awesome brilliant superb perfect nice lovely fun enjoy positive wonderful terrific
fabulous outstanding marvelous splendid delightful pleasant glad pleased cheerful
""".split())

NEGATIVE_WORDS = set("""
bad terrible awful horrible disgusting hate awful worst poor ugly sad angry
disappointing dreadful horrible unpleasant annoying frustrated terrible lousy
pathetic miserable horrible depressing awful horrible bad negative terrible
""".split())

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

def rule_sentiment(text: str) -> tuple:
    tokens = set(text.lower().split())
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
    score = pos - neg
    if score > 0:   return "Positive", score
    elif score < 0: return "Negative", score
    else:           return "Neutral", 0

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv("../../datasets/sentiment.csv", index_col=0)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
df.columns = df.columns.str.strip()
df["Text"]      = df["Text"].astype(str).str.strip()
df["Sentiment"] = df["Sentiment"].astype(str).str.strip()

print("=" * 55)
print("LEVEL 3 – TASK 3: NLP SENTIMENT ANALYSIS")
print("=" * 55)
print(f"\nDataset shape  : {df.shape}")
print(f"Columns        : {list(df.columns)}")
print(f"\nSentiment distribution (original labels):")
print(df["Sentiment"].value_counts())

# ── 2. Preprocess Text ───────────────────────────────────────
df["clean_text"] = df["Text"].apply(preprocess)
print("\n[2] Text preprocessed (lowercased, URLs/mentions/punctuation removed)")
print("\n   Sample transformations:")
for _, row in df.head(3).iterrows():
    print(f"   Original : {row['Text'][:65].strip()}")
    print(f"   Cleaned  : {row['clean_text'][:65]}\n")

# ── 3. Rule-based Sentiment Scoring ─────────────────────────
df[["rule_sentiment","rule_score"]] = df["clean_text"].apply(
    lambda t: pd.Series(rule_sentiment(t)))

print("[3] Rule-based sentiment scored")
print(f"\n   Rule-based distribution:")
print(df["rule_sentiment"].value_counts())

# Agreement
common_labels = df[df["Sentiment"].isin(["Positive","Negative","Neutral"])]
if len(common_labels) > 0:
    match = (common_labels["Sentiment"] == common_labels["rule_sentiment"]).mean()
    print(f"\n   Label agreement (original vs rule-based): {match:.2%}")

# ── 4. Sentiment Distribution ────────────────────────────────
COLORS = {"Positive":"#2ecc71","Negative":"#e74c3c","Neutral":"#3498db"}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Sentiment Analysis Results", fontsize=15, fontweight="bold")

for ax, col, title in [
    (axes[0], "Sentiment",      "Original Labels"),
    (axes[1], "rule_sentiment", "Rule-Based Prediction"),
]:
    counts = df[col].value_counts()
    bar_colors = [COLORS.get(s, "#95a5a6") for s in counts.index]
    bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                str(val), ha="center", fontweight="bold", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig("plots/sentiment_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[4] Sentiment distribution saved → plots/sentiment_distribution.png")

# ── 5. Word Frequency Analysis ───────────────────────────────
# Simulate polarity via score for histogram
df["polarity_approx"] = df["rule_score"] / (df["rule_score"].abs().max() + 1)

fig, ax = plt.subplots(figsize=(10, 5))
for sent, color in COLORS.items():
    sub = df[df["rule_sentiment"] == sent]["polarity_approx"]
    if len(sub) > 0:
        ax.hist(sub, bins=20, alpha=0.6, color=color, label=sent, edgecolor="white")
ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
ax.set_title("Sentiment Score Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Sentiment Score (Negative ← 0 → Positive)")
ax.set_ylabel("Frequency"); ax.legend(); ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("plots/polarity_histogram.png", dpi=150, bbox_inches="tight")
plt.close()
print("[5] Polarity histogram saved → plots/polarity_histogram.png")

# ── 6. Word Cloud (text-based bar chart substitute) ──────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Top 20 Words per Sentiment Category", fontsize=15, fontweight="bold")
for ax, (sent, color) in zip(axes, COLORS.items()):
    sub_text = " ".join(df[df["Sentiment"].str.strip()==sent]["clean_text"].dropna())
    words = sub_text.split()
    freq  = Counter(words).most_common(20)
    if freq:
        wds, cnts = zip(*freq)
        ax.barh(list(wds)[::-1], list(cnts)[::-1], color=color, edgecolor="white")
    ax.set_title(f"{sent} – Top Words", fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency"); ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
plt.savefig("plots/top_words.png", dpi=150, bbox_inches="tight")
plt.close()
print("[6] Top words plot saved → plots/top_words.png")

# ── 7. Platform Analysis ─────────────────────────────────────
if "Platform" in df.columns:
    plat = df["Platform"].str.strip()
    plat_sentiment = df.groupby(plat)["Sentiment"].value_counts(normalize=True).unstack(fill_value=0)
    if set(COLORS.keys()).intersection(plat_sentiment.columns):
        fig, ax = plt.subplots(figsize=(12, 5))
        plat_sentiment[[c for c in COLORS if c in plat_sentiment.columns]].plot(
            kind="bar", ax=ax,
            color=[COLORS[c] for c in COLORS if c in plat_sentiment.columns],
            edgecolor="white", alpha=0.85)
        ax.set_title("Sentiment Distribution by Platform", fontsize=14, fontweight="bold")
        ax.set_xlabel("Platform"); ax.set_ylabel("Proportion")
        ax.legend(title="Sentiment"); ax.grid(axis="y", alpha=0.4)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("plots/platform_sentiment.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[7] Platform sentiment saved → plots/platform_sentiment.png")

# ── 8. Country Analysis ──────────────────────────────────────
if "Country" in df.columns:
    top_countries = df["Country"].str.strip().value_counts().head(8).index
    sub = df[df["Country"].str.strip().isin(top_countries)]
    country_sent = sub.groupby(sub["Country"].str.strip())["Sentiment"].value_counts(
        normalize=True).unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    country_sent[[c for c in COLORS if c in country_sent.columns]].plot(
        kind="bar", ax=ax,
        color=[COLORS[c] for c in COLORS if c in country_sent.columns],
        edgecolor="white", alpha=0.85)
    ax.set_title("Sentiment Distribution by Country (Top 8)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Country"); ax.set_ylabel("Proportion")
    ax.legend(title="Sentiment"); ax.grid(axis="y", alpha=0.4)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("plots/country_sentiment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[8] Country sentiment saved → plots/country_sentiment.png")

# ── 9. Temporal Trend ────────────────────────────────────────
if "Year" in df.columns and "Month" in df.columns:
    df["period"] = df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2)
    trend = df.groupby(["period","Sentiment"]).size().unstack(fill_value=0)
    if len(trend) > 1:
        fig, ax = plt.subplots(figsize=(14, 5))
        for sent, color in COLORS.items():
            if sent in trend.columns:
                ax.plot(trend.index, trend[sent], marker="o", color=color,
                        linewidth=2, label=sent)
        ax.set_title("Sentiment Trend Over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel("Period"); ax.set_ylabel("Count")
        ax.legend(); plt.xticks(rotation=45); ax.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig("plots/sentiment_trend.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[9] Sentiment trend saved → plots/sentiment_trend.png")

print("\n✅  Task 3 complete!\n")
print("NOTE: To run with full NLTK + TextBlob support in your local environment:")
print("      pip install nltk textblob wordcloud")
print("      Then replace rule_sentiment() with TextBlob().sentiment.polarity")
