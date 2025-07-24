# 📱 Telecom Customer Churn Prediction App

This project aims to predict whether a telecom customer will **Churn**, **Stay**, or **Join**, based on various factors such as tenure, charges, internet type, contract type, and more.

## 🎯 Objective

The goal is to help telecom companies identify which customers are at risk of leaving (churning), so they can take proactive steps like targeted offers, improved service, or loyalty programs.

---

## 📊 Features Used for Prediction

- Age  
- Number of Dependents  
- Zip Code  
- Number of Referrals  
- Tenure in Months  
- Avg Monthly Long Distance Charges  
- Multiple Lines  
- Avg Monthly GB Download  
- Online Security  
- Premium Tech Support  
- Monthly Charge  
- Total Charges  
- Total Extra Data Charges  
- Total Long Distance Charges  
- Total Revenue  
- Offer (Offer E)  
- Internet Type (DSL, Fiber Optic)  
- Contract (Month-to-Month, One Year, Two Year)  
- Payment Method (Credit Card)

---

## 🧠 Machine Learning Model

- Model Used: `XGBoost Classifier`
- Classification Targets: 
  - **0** - Churned  
  - **1** - Joined  
  - **2** - Stayed

We also used a Random Forest Classifier to identify the **top 10 important features** influencing churn.

---

## 📈 Metrics & Evaluation

- **Accuracy**: Measures how many predictions were correct.
- **Precision**: How many predicted churns were actually correct.
- **Recall**: How many actual churns we successfully caught.
- **F1 Score**: Balance between Precision and Recall.
- **ROC Curve**: Visual tool to show model’s ability to distinguish between classes (ideal = closer to top-left).

These metrics help the business:
- Understand how reliable the model is.
- Identify which group of customers need more focus.
- Take data-driven actions to retain high-value users.

---

## 🚀 How to Run the App (Locally)

1. **Clone the Repository**  
   ```bash
   git clone <repo-url>
   cd telecom-churn-app

2. python -m venv .venv
    source .venv/bin/activate  # Linux/Mac
    .venv\Scripts\activate     # Windows

3. pip install -r requirements.txt

4. streamlit run app.py
