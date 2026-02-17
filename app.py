import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Churn AI", page_icon="🔮", layout="centered")

MODEL_PATH = "model/churn_pipeline.pkl"
DATA_PATH = "data/Telco-Customer-Churn.csv"

@st.cache_resource
def load_or_train():
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle
    except Exception:
        st.warning("Model couldn't be loaded. Training a fresh model on the server (one-time)...")

        from sklearn.model_selection import train_test_split
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import f1_score
        from xgboost import XGBClassifier
        import numpy as np

        df = pd.read_csv(DATA_PATH)
        df.drop("customerID", axis=1, inplace=True)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

        X = df.drop("Churn", axis=1)
        y = df["Churn"].map({"Yes": 1, "No": 0})

        cat_cols = X.select_dtypes(include="object").columns.tolist()
        num_cols = X.select_dtypes(exclude="object").columns.tolist()

        preprocess = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ])

        scale_pos_weight = (y == 0).sum() / (y == 1).sum()

        model = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )

        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.23, stratify=y, random_state=42
        )

        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        best_t, best_f1 = 0.5, 0
        for t in [0.35, 0.40, 0.45, 0.50]:
            y_pred = (proba > t).astype(int)
            f1 = f1_score(y_test, y_pred)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        os.makedirs("model", exist_ok=True)
        joblib.dump({"pipeline": pipe, "threshold": best_t}, MODEL_PATH)

        return {"pipeline": pipe, "threshold": best_t}

bundle = load_or_train()
pipe = bundle["pipeline"]
threshold = bundle["threshold"]

# ---- UI ----
st.title("🔮 Customer Churn Predictor")
st.caption("Predict if a customer is likely to churn using ML")

gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
tenure = st.slider("Tenure (months)", 0, 72, 6)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=5.0)
tech_support = st.toggle("Has Tech Support?")
online_security = st.toggle("Has Online Security?")
paperless = st.toggle("Paperless Billing?")

input_data = {
    "gender": gender,
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": tenure,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "Yes" if online_security else "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "Yes" if tech_support else "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": contract,
    "PaperlessBilling": "Yes" if paperless else "No",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": monthly,
    "TotalCharges": round(monthly * max(tenure, 1), 2)
}

df_input = pd.DataFrame([input_data])

if st.button("🚀 Predict Churn", use_container_width=True):
    proba = pipe.predict_proba(df_input)[0][1]
    pred = 1 if proba > threshold else 0

    if pred == 1:
        st.error(f"❌ Likely to CHURN ({proba*100:.1f}%)")
        st.progress(int(proba * 100))
        st.write("💡 Tip: Offer discounts, move to yearly plan, or proactive support.")
    else:
        st.success(f"✅ Likely to STAY ({(1-proba)*100:.1f}%)")
        st.progress(int((1 - proba) * 100))
        st.write("🎉 Tip: Maintain service quality & loyalty rewards.")

st.caption("Deployed with Streamlit • ML Pipeline + Threshold Tuning")
