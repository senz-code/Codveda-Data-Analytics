# 🚀 GitHub Setup Guide – Codveda Data Analytics

Follow these steps to upload your project to GitHub.

---

## Step 1: Create a GitHub Account
Go to https://github.com and sign up (if you don't have an account).

---

## Step 2: Create a New Repository

1. Click the **"+"** icon → **"New repository"**
2. Repository name: `codveda-data-analytics`
3. Description: `Codveda Technology Data Analytics Internship – All 3 Levels`
4. Set to **Public**
5. ✅ Check "Add a README file" → **NO** (we have our own)
6. Click **"Create repository"**

---

## Step 3: Install Git (if not already installed)

**Windows:**
- Download from https://git-scm.com/download/win
- Run installer, keep all defaults

**Mac:**
```bash
brew install git
```

**Linux (Ubuntu):**
```bash
sudo apt install git
```

Verify: `git --version`

---

## Step 4: Configure Git (first time only)

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

---

## Step 5: Clone & Push Your Project

```bash
# 1. Navigate to your project folder
cd path/to/codveda-data-analytics

# 2. Initialise git
git init

# 3. Add all files
git add .

# 4. First commit
git commit -m "Initial commit: All 3 levels of Codveda Data Analytics tasks"

# 5. Connect to GitHub (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/codveda-data-analytics.git

# 6. Push to GitHub
git branch -M main
git push -u origin main
```

---

## Step 6: Verify on GitHub

Visit: `https://github.com/YOUR_USERNAME/codveda-data-analytics`

You should see all your files and the README rendered beautifully.

---

## Step 7: Update Your LinkedIn Post

Copy and paste this template:

```
🎉 Excited to share my Data Analytics internship project with @Codveda Technology!

Over the past weeks, I completed all 3 levels of the Data Analytics track:

✅ Level 1 (Basic):
  • Data Cleaning & Preprocessing (Iris dataset)
  • Exploratory Data Analysis with visualisations
  • Basic Data Visualisation (bar, line, scatter plots)

✅ Level 2 (Intermediate):
  • Linear Regression on House Price data (R²=0.67)
  • Time Series Analysis on Stock Prices (AAPL)
  • K-Means Clustering with PCA visualisation

✅ Level 3 (Advanced):
  • Customer Churn Prediction (Random Forest, F1=0.81)
  • Multi-dataset Analytics Dashboard
  • NLP Sentiment Analysis on Social Media data

🔗 GitHub: https://github.com/YOUR_USERNAME/codveda-data-analytics

#CodvedaJourney #CodvedaExperience #FutureWithCodveda
#DataAnalytics #Python #MachineLearning #NLP
```

---

## Pro Tips

- **Commit often:** after each task, `git add . && git commit -m "Complete Level X Task Y"`
- **Add plots:** your generated PNG files in `plots/` folders will show in GitHub
- **Pin the repo:** on your GitHub profile, pin this repository so it's visible
- **README matters:** the README.md file renders on your repo homepage — keep it polished
