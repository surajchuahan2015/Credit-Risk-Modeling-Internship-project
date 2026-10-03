import sqlite3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "credit_risk.csv"
OUT = BASE / "outputs"
MODELS = BASE / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Basic EDA
plt.figure(figsize=(7,5))
sns.countplot(data=df, x="default")
plt.title("Loan Default Distribution")
plt.tight_layout()
plt.savefig(OUT / "default_distribution.png")
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(data=df, x="credit_score", hue="default", bins=10, kde=True)
plt.title("Credit Score vs Default")
plt.tight_layout()
plt.savefig(OUT / "credit_score_vs_default.png")
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="income", y="loan_amount", hue="default", s=80)
plt.title("Income vs Loan Amount")
plt.tight_layout()
plt.savefig(OUT / "income_vs_loan.png")
plt.close()

X = df.drop(columns=["loan_id", "default"])
y = df["default"]

numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical = X.select_dtypes(include=["object"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric),
    ("cat", categorical_pipe, categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
}

results = []

for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, prob)
    })

    print(f"\n{name}")
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

    if name == "Random Forest":
        joblib.dump(pipe, MODELS / "random_forest_credit_risk.joblib")

results_df = pd.DataFrame(results)
print("\nModel comparison:\n", results_df.to_string(index=False))
results_df.to_csv(OUT / "model_comparison.csv", index=False)

plt.figure(figsize=(8,5))
results_melt = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=results_melt, x="Metric", y="Score", hue="Model")
plt.ylim(0, 1.05)
plt.title("Model Evaluation Comparison")
plt.tight_layout()
plt.savefig(OUT / "model_comparison.png")
plt.close()

# SQL demonstration using SQLite
db = OUT / "credit_risk.db"
conn = sqlite3.connect(db)
df.to_sql("credit_risk", conn, if_exists="replace", index=False)
query = """
SELECT "default",
       COUNT(*) AS applicants,
       ROUND(AVG(credit_score), 2) AS avg_credit_score,
       ROUND(AVG(debt_to_income), 3) AS avg_dti
FROM credit_risk
GROUP BY "default"
"""
sql_result = pd.read_sql_query(query, conn)
sql_result.to_csv(OUT / "sql_summary.csv", index=False)
conn.close()

print("\nProject completed successfully.")
print("Charts:", OUT)
print("Model:", MODELS / "random_forest_credit_risk.joblib")
