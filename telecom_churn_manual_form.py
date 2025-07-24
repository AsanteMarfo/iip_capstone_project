import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="https://www.svgrepo.com/show/494506/customer-service-communication-customer-service-sister-paper.svg",
    layout="centered", # Can be "wide" or "centered"
    initial_sidebar_state="expanded"
)

# --- Load Model and Selected Features (Cached for Performance) ---
@st.cache_resource
def load_resources():
    try:
        with open("xgboost_classifier_only.pkl", "rb") as f:
            model = pickle.load(f)
        with open("selected_features.pkl", "rb") as f:
            selected_features = pickle.load(f)
        return model, selected_features
    except FileNotFoundError:
        st.error("Error: Model or selected features file not found. Please ensure 'xgboost_classifier_only.pkl' and 'selected_features.pkl' are in the same directory.")
        st.stop() # Stop the app if files are missing

model, selected_features = load_resources()

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #FF6347; /* Tomato Red */
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 5px #ccc;
    }
    .stSelectbox, .stSlider, .stTextInput {
        margin-bottom: 15px;
    }
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-size: 1.1em;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .prediction-result {
        font-size: 1.8em;
        color: #1E90FF; /* Dodger Blue */
        text-align: center;
        margin-top: 30px;
        font-weight: bold;
    }
    .probability-item {
        font-size: 1.1em;
        margin-bottom: 8px;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6; /* Light gray for sidebar */
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<h1 class="main-header">📱 Telecom Customer Churn Prediction</h1>', unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
menu_selection = st.sidebar.radio("Go to", ["Overview", "Prediction", "Top Features"])

# --- Page Content ---
if menu_selection == "Overview":
    st.header("About This Application")
    st.write("""
        Welcome to the **Telecom Customer Churn Prediction** application!
        This tool helps telecommunication companies predict whether a customer is likely to churn (leave their service), join, or stay.
        By inputting various customer details, the underlying machine learning model will provide a prediction and the probabilities associated with each outcome.

        Understanding customer churn is crucial for businesses to implement retention strategies, improve customer satisfaction, and optimize their services.
        This application leverages an **XGBoost Classifier** model, trained on historical telecom customer data.
    """)
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSAoNYV0AosCQ1X418yumqdXgWDYlLTvMjZUg&s", caption="Customer Churn Prediction", use_container_width=True)
    st.write("""
        ---
        ### How it Works:
        1. **Navigate to 'PREDICTION'**: Use the sidebar to go to the prediction page.
        2. **Enter Customer Details**: Fill in the various features related to a customer.
        3. **Get Prediction**: Click the 'Predict Churn' button to see the model's prediction and confidence levels.
        4. **Explore 'TOP FEATURES'**: Understand which customer attributes are most influential in the churn prediction.
    """)

elif menu_selection == "Prediction":
    st.header("Predict Customer Churn")
    st.markdown("Fill in the customer's details below to predict churn status.")

    # === Manual Input Fields Matching Your Feature List ===
    st.subheader("Customer Demographics & Service Usage")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 35)
        num_dependents = st.slider("Number of Dependents", 0, 10, 0)
        num_referrals = st.slider("Number of Referrals", 0, 10, 0)
        tenure_months = st.slider("Tenure in Months", 0, 72, 12)
        monthly_charge = st.slider("Monthly Charge ($)", 0.0, 200.0, 70.0)
    with col2:
        total_charges = st.slider("Total Charges ($)", 0.0, 10000.0, 2000.0)
        total_revenue = st.slider("Total Revenue ($)", 0.0, 15000.0, 3000.0)
        avg_long_dist = st.slider("Avg Monthly Long Distance Charges ($)", 0.0, 100.0, 10.0)
        long_dist_charges = st.slider("Total Long Distance Charges ($)", 0.0, 1000.0, 100.0)
        extra_data_charges = st.slider("Total Extra Data Charges ($)", 0.0, 500.0, 0.0)

    st.subheader("Service Specifics")
    col3, col4 = st.columns(2)
    with col3:
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"], help="Does the customer have multiple phone lines?")
        online_security = st.selectbox("Online Security", ["Yes", "No"], help="Does the customer subscribe to online security services?")
        premium_tech_support = st.selectbox("Premium Tech Support", ["Yes", "No"], help="Does the customer have premium tech support?")
    with col4:
        avg_gb_download = st.slider("Avg Monthly GB Download", 0.0, 100.0, 10.0, help="Average Gigabytes downloaded per month.")
        internet_type = st.selectbox("Internet Type", ["None", "DSL", "Fiber Optic", "Cable"], help="Type of internet service.")
        offer = st.selectbox("Offer", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"], help="Any special offers the customer is on.")

    st.subheader("Contract & Payment")
    col5, col6 = st.columns(2)
    with col5:
        contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"], help="Customer's contract type.")
    with col6:
        payment_method = st.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"], help="How the customer pays their bills.")

    # === Preprocessing Inputs ===
    input_dict = {
        'Age': age,
        'Number of Dependents': num_dependents,
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

    # One-hot encoded fields (make sure these align with your model's training)
    # Streamlit offers a clean way to handle these with columns for better layout
    offer_options = ["Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]
    for val in offer_options:
        input_dict[f"Offer_{val}"] = 1 if offer == val else 0

    internet_type_options = ["DSL", "Fiber Optic", "Cable"]
    for val in internet_type_options:
        input_dict[f"Internet Type_{val}"] = 1 if internet_type == val else 0

    contract_options = ["Month-to-Month", "One Year", "Two Year"]
    for val in contract_options:
        input_dict[f"Contract_{val}"] = 1 if contract == val else 0

    payment_method_options = ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"]
    for val in payment_method_options:
        input_dict[f"Payment Method_{val}"] = 1 if payment_method == val else 0

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Ensure all features are present and in the correct order as per selected_features
    for feat in selected_features:
        if feat not in input_df.columns:
            input_df[feat] = 0

    input_df = input_df[selected_features] # Reorder columns

    # === Prediction ===
    st.markdown("---")
    if st.button("Predict Churn"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        status_map = {0: "Churned", 1: "Joined", 2: "Stayed"}
        predicted_status = status_map.get(prediction, "Unknown")

        st.markdown(f'<p class="prediction-result">Prediction: {predicted_status}</p>', unsafe_allow_html=True)
        st.write("### Probabilities:")
        # Sort probabilities for better readability (highest first)
        proba_df = pd.DataFrame({'Status': ["Churned", "Joined", "Stayed"], 'Probability': proba})
        proba_df = proba_df.sort_values(by='Probability', ascending=False)

        for _, row in proba_df.iterrows():
            st.markdown(f'<p class="probability-item">- **{row["Status"]}**: <span style="font-weight: bold; color: #DC143C;">{row["Probability"]:.2%}</span></p>', unsafe_allow_html=True)

elif menu_selection == "Top Features":
    st.header("Top Influencing Features")
    st.write("""
        This section highlights the most important features that the model uses to make its predictions.
        Understanding these features can provide insights into what drives customer churn.
    """)

    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.Series(model.feature_importances_, index=selected_features)
        top_n = st.slider("Show Top N Features", 5, len(selected_features), 10)
        top_features = feature_importances.nlargest(top_n)

        st.subheader(f"Top {top_n} Features by Importance:")
        st.bar_chart(top_features)

        st.write("""
            **Interpretation:**
            * **Longer bars** indicate a higher importance, meaning that changes in these features have a greater impact on the model's prediction of churn, join, or stay.
            * This can help in identifying key areas for business intervention.
        """)
    else:
        st.warning("Feature importances are not available for this model type or have not been calculated.")

st.markdown("---")
st.info("💡 This is a demo application. Predictions are based on a pre-trained model and may not reflect real-world scenarios perfectly.")
