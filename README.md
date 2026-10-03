# Credit Risk Modeling — Data Analytics Internship

## Project Overview
This project predicts whether a loan applicant is likely to default using data analytics and machine learning.

It is designed around the internship topics:
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Advanced Data Analysis
- Machine Learning Basics
- Applied Machine Learning
- SQL data analysis
- Model evaluation and interpretation

## Business Problem
Financial institutions need to identify applicants with higher credit risk while using applicant information responsibly. The model predicts the `default` target from financial and demographic features.

## Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- SQL
- SQLite
- Joblib

## Project Structure
```text
Credit_Risk_Modeling_Internship_Project/
├── data/
│   └── credit_risk.csv
├── sql/
│   └── credit_risk_queries.sql
├── outputs/
├── models/
├── credit_risk_analysis.py
├── requirements.txt
└── README.md
```

## SQL Work
The SQL file demonstrates:
- SELECT and filtering
- GROUP BY
- ORDER BY
- CASE statements
- Aggregation
- Default-rate analysis

## Python/ML Work
1. Load and inspect the dataset
2. Clean and validate data
3. Perform EDA
4. Encode categorical variables
5. Split training and test data
6. Train Logistic Regression and Random Forest
7. Compare Accuracy, Precision, Recall and ROC-AUC
8. Save the trained model
9. Save charts in `outputs/`

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the analysis
```bash
python credit_risk_analysis.py
```

The charts will be saved in `outputs/` and the trained model in `models/`.

## SQL
SQLite is used so no separate database server is required. The SQL queries can be opened in SQLite, DB Browser for SQLite, or adapted to MySQL/PostgreSQL.

## Dataset
A small synthetic dataset is included for learning and demonstration. For an extended internship submission, it can be replaced with a larger public lending dataset.

## Disclaimer
This is an educational data-science project. It is not a production credit decision system. Real lending decisions require validated models, appropriate governance, fairness checks, privacy controls, and regulatory compliance.
