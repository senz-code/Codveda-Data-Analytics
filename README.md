# 📊 Codveda Technology – Data Analytics Internship

> **Intern:** [Your Name]
> **Domain:** Data Analytics
> **Duration:** 1 Month
> **Submission:** Codveda Submission Form

---

## 🗂️ Repository Structure

```
codveda-data-analytics/
│
├── datasets/                        # All raw CSV datasets
│   ├── iris.csv
│   ├── stock_prices.csv
│   ├── sentiment.csv
│   ├── house_prices.csv
│   ├── churn_train.csv
│   └── churn_test.csv
│
├── Level_1/                         # Basic Tasks
│   ├── Task1_Data_Cleaning/
│   │   └── task1_data_cleaning.py
│   ├── Task2_EDA/
│   │   └── task2_eda.py
│   └── Task3_Visualization/
│       └── task3_visualization.py
│
├── Level_2/                         # Intermediate Tasks
│   ├── Task1_Regression/
│   │   └── task1_regression.py
│   ├── Task2_TimeSeries/
│   │   └── task2_time_series.py
│   └── Task3_Clustering/
│       └── task3_clustering.py
│
├── Level_3/                         # Advanced Tasks
│   ├── Task1_Classification/
│   │   └── task1_classification.py
│   ├── Task2_Dashboard/
│   │   └── task2_dashboard.py
│   └── Task3_NLP/
│       └── task3_nlp_sentiment.py
│
├── requirements.txt
└── README.md
```

---

## 📋 Task Overview

### 🟢 Level 1 – Basic

| Task | Description | Dataset | Tools |
|------|-------------|---------|-------|
| Task 1 | Data Cleaning & Preprocessing | Iris | pandas, numpy |
| Task 2 | Exploratory Data Analysis (EDA) | Iris | pandas, matplotlib, seaborn |
| Task 3 | Basic Data Visualization | Iris | matplotlib, seaborn |

### 🟡 Level 2 – Intermediate

| Task | Description | Dataset | Tools |
|------|-------------|---------|-------|
| Task 1 | Regression Analysis | House Prices | scikit-learn, pandas |
| Task 2 | Time Series Analysis | Stock Prices | statsmodels, matplotlib |
| Task 3 | Clustering Analysis (K-Means) | Iris | scikit-learn, seaborn |

### 🔴 Level 3 – Advanced

| Task | Description | Dataset | Tools |
|------|-------------|---------|-------|
| Task 1 | Predictive Modeling (Classification) | Churn | scikit-learn, matplotlib |
| Task 2 | Interactive Dashboard | Churn + Stocks + Iris | Plotly |
| Task 3 | NLP – Sentiment Analysis | Sentiment | nltk, TextBlob, wordcloud |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/codveda-data-analytics.git
cd codveda-data-analytics
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Any Task
Navigate to the task folder and run the Python script:

```bash
# Example – Level 1, Task 1
cd Level_1/Task1_Data_Cleaning
python task1_data_cleaning.py

# Example – Level 2, Task 2
cd Level_2/Task2_TimeSeries
python task2_time_series.py

# Example – Level 3, Task 3
cd Level_3/Task3_NLP
python task3_nlp_sentiment.py
```

Each script:
- Prints results to the console
- Saves plots to a `plots/` subfolder inside the task directory
- Saves any output files (cleaned CSVs, HTML dashboards) locally

---

## 📈 Key Results Summary

### Level 1
- **Data Cleaning:** Handled missing values via median imputation, removed duplicates, standardised categorical formats
- **EDA:** Discovered strong positive correlation (0.96) between petal length and petal width in the Iris dataset
- **Visualisation:** Created bar plots, line charts, scatter plots, and a summary dashboard

### Level 2
- **Regression:** Linear Regression on Boston Housing achieved **R² = ~0.74**
- **Time Series:** Decomposed AAPL stock into trend, seasonality and residuals; performed 7/30/90-day moving averages
- **Clustering:** K-Means (K=3) on Iris achieved **Silhouette Score ≈ 0.55** and **ARI ≈ 0.73**

### Level 3
- **Classification:** Random Forest achieved **F1-score ≈ 0.92** on churn prediction after GridSearchCV tuning
- **Dashboard:** Interactive Plotly HTML dashboard with zoom, filters, and hover tooltips
- **NLP:** TextBlob sentiment analysis with word clouds and temporal trend visualisation

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **Data Manipulation:** pandas, numpy
- **Visualisation:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn
- **Time Series:** statsmodels
- **NLP:** nltk, TextBlob, wordcloud

---

## 📬 Contact

- **Company:** Codveda Technology
- **LinkedIn:** [@codveda](https://linkedin.com/company/codveda)
- **Email:** support@codveda.com
- **Website:** [www.codveda.com](https://www.codveda.com)

---

*#CodvedaJourney #CodvedaExperience #FutureWithCodveda*
