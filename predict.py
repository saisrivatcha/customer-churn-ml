import joblib
import pandas as pd

bundle = joblib.load("model/churn_pipeline.pkl")
pipe = bundle["pipeline"]
threshold = bundle["threshold"]

# Raw input (strings allowed now!)
input_data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 6,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 513.0
}

df_input = pd.DataFrame([input_data])

proba = pipe.predict_proba(df_input)[0][1]
pred = 1 if proba > threshold else 0

print("Prediction:", "CHURN ❌" if pred == 1 else "STAY ✅")
print("Churn Probability:", round(proba * 100, 2), "%")
