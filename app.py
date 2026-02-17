import streamlit as st
import joblib
import pandas as pd

bundle = joblib.load("model/churn_pipeline.pkl")
pipe = bundle["pipeline"]
threshold = bundle["threshold"]

st.title("🔮 Customer Churn Predictor")

gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
tenure = st.slider("Tenure (months)", 0, 72, 6)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
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

if st.button("Predict Churn"):
    proba = pipe.predict_proba(df_input)[0][1]
    pred = 1 if proba > threshold else 0

    if pred == 1:
        st.error(f"❌ Likely to CHURN ({proba*100:.1f}%)")
    else:
        st.success(f"✅ Likely to STAY ({(1-proba)*100:.1f}%)")
