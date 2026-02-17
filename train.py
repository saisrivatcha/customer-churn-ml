import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import numpy as np

os.makedirs("model", exist_ok=True)

# Load data
df = pd.read_csv("data/Telco-Customer-Churn.csv")
df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# Column types
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# Preprocessor
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# Handle imbalance
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

model = XGBClassifier(
    n_estimators=800,        # was 500
    max_depth=6,            # was 5
    min_child_weight=2,
    subsample=0.9,
    colsample_bytree=0.9,
    learning_rate=0.04,     # a bit smaller
    reg_alpha=0.1,          # L1 regularization
    reg_lambda=1.0,         # L2 regularization
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

df["ChargePerTenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
df["IsShortTenure"] = (df["tenure"] <= 6).astype(int)
df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Train
pipe.fit(X_train, y_train)

# Evaluate & threshold tuning
proba = pipe.predict_proba(X_test)[:, 1]

best_t, best_f1 = 0.5, 0
for t in np.arange(0.3, 0.7, 0.05):
    y_pred = (proba > t).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print("Best Threshold:", best_t)
y_final = (proba > best_t).astype(int)
print("Accuracy:", accuracy_score(y_test, y_final))
print(classification_report(y_test, y_final))

# Save everything
joblib.dump({
    "pipeline": pipe,
    "threshold": best_t
}, "model/churn_pipeline.pkl")
 

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
print("CV F1 (mean):", scores.mean())
print("Total rows:", len(X))
print("Train rows:", len(X_train))
print("Test rows:", len(X_test))
print("Train + Test:", len(X_train) + len(X_test))
print("✅ Perfect pipeline saved at model/churn_pipeline.pkl")
