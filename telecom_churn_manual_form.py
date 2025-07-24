import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and selected features
with open("xgboost_classifier_only.pkl", "rb") as f:
    model = pickle.load(f)

with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

st.title("📱 Telecom Customer Churn Prediction")
st.markdown("Fill in the customer's details below to predict churn status.")

# === Manual Input Fields Matching Your Feature List ===
age = st.slider("Age", 18, 100, 35)
num_dependents = st.slider("Number of Dependents", 0, 10, 0)
# zip_code = st.text_input("Zip Code", "12345")
num_referrals = st.slider("Number of Referrals", 0, 10, 0)
tenure_months = st.slider("Tenure in Months", 0, 72, 12)
avg_long_dist = st.slider("Avg Monthly Long Distance Charges", 0.0, 100.0, 10.0)
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
avg_gb_download = st.slider("Avg Monthly GB Download", 0.0, 100.0, 10.0)
online_security = st.selectbox("Online Security", ["Yes", "No"])
premium_tech_support = st.selectbox("Premium Tech Support", ["Yes", "No"])
monthly_charge = st.slider("Monthly Charge", 0.0, 200.0, 70.0)
total_charges = st.slider("Total Charges", 0.0, 10000.0, 2000.0)
extra_data_charges = st.slider("Total Extra Data Charges", 0.0, 500.0, 0.0)
long_dist_charges = st.slider("Total Long Distance Charges", 0.0, 1000.0, 100.0)
total_revenue = st.slider("Total Revenue", 0.0, 15000.0, 3000.0)

# One-hot encodable features
offer = st.selectbox("Offer", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])
internet_type = st.selectbox("Internet Type", ["None", "DSL", "Fiber Optic", "Cable"])
contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
payment_method = st.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"])

# === Preprocessing Inputs ===
input_dict = {
    'Age': age,
    'Number of Dependents': num_dependents,
    # 'Zip Code': zip_code,
    'Number of Referrals': num_referrals,
    'Tenure in Months': tenure_months,
    'Avg Monthly Long Distance Charges': avg_long_dist,
    'Multiple Lines': 1 if multiple_lines == "Yes" else 0,
    'Avg Monthly GB Download': avg_gb_download,
    'Online Security': 1 if online_security == "Yes" else 0,
    'Premium Tech Support': 1 if premium_tech_support == "Yes" else 0,
    'Monthly Charge': monthly_charge,
    'Total Charges': total_charges,
    'Total Extra Data Charges': extra_data_charges,
    'Total Long Distance Charges': long_dist_charges,
    'Total Revenue': total_revenue,
}

# One-hot encoded fields
for val in ["Offer E"]:
    input_dict[f"Offer_{val}"] = 1 if offer == val else 0

for val in ["DSL", "Fiber Optic"]:
    input_dict[f"Internet Type_{val}"] = 1 if internet_type == val else 0

for val in ["Month-to-Month", "One Year", "Two Year"]:
    input_dict[f"Contract_{val}"] = 1 if contract == val else 0

for val in ["Credit Card"]:
    input_dict[f"Payment Method_{val}"] = 1 if payment_method == val else 0

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Ensure all features are present
for feat in selected_features:
    if feat not in input_df.columns:
        input_df[feat] = 0

# Reorder columns
input_df = input_df[selected_features]

# === Prediction ===
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    status_map = {0: "Churned", 1: "Joined", 2: "Stayed"}
    predicted_status = status_map.get(prediction, "Unknown")

    st.subheader(f"Prediction: {predicted_status}")
    st.write("### Probabilities:")
    for label, prob in zip(["Churned", "Joined", "Stayed"], proba):
        st.write(f"- {label}: {prob:.2%}")
