import streamlit as st
import joblib
import numpy as np

# Load models
lr_model = joblib.load('lr_model.pkl')
gbr_model = joblib.load('gbr_model.pkl')
svr_lin_model = joblib.load('svr_lin_model.pkl')

# Streamlit app
st.title("Customer Lifetime Value (LTV) Prediction App")
st.write("Enter the customer details below to predict their LTV:")

# Input fields for all 27 features
customer_id = st.text_input("Customer ID")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
location = st.text_input("Location")
income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
total_transactions = st.number_input("Total Transactions", min_value=0, value=5)
avg_transaction_value = st.number_input("Average Transaction Value", min_value=0.0, value=100.0)
max_transaction_value = st.number_input("Max Transaction Value", min_value=0.0, value=200.0)
min_transaction_value = st.number_input("Min Transaction Value", min_value=0.0, value=10.0)
total_spent = st.number_input("Total Spent", min_value=0.0, value=500.0)
active_days = st.number_input("Active Days", min_value=0, value=100)
last_transaction_days_ago = st.number_input("Last Transaction Days Ago", min_value=0, value=10)
loyalty_points_earned = st.number_input("Loyalty Points Earned", min_value=0, value=200)
referral_count = st.number_input("Referral Count", min_value=0, value=2)
cashback_received = st.number_input("Cashback Received", min_value=0.0, value=50.0)
app_usage_frequency = st.number_input("App Usage Frequency (per week)", min_value=0, value=5)
preferred_payment_method = st.selectbox("Preferred Payment Method", ["Credit Card", "Debit Card", "Net Banking", "UPI", "Wallet"])
support_tickets_raised = st.number_input("Support Tickets Raised", min_value=0, value=1)
issue_resolution_time = st.number_input("Issue Resolution Time (hours)", min_value=0.0, value=5.0)
customer_satisfaction_score = st.slider("Customer Satisfaction Score", 0.0, 10.0, 7.5)

# Additional fields to complete 27 features
avg_session_duration = st.number_input("Average Session Duration (mins)", min_value=0.0, value=15.0)
times_logged_in_last_month = st.number_input("Times Logged In Last Month", min_value=0, value=12)
promotions_availed = st.number_input("Promotions Availed", min_value=0, value=3)
returns_made = st.number_input("Returns Made", min_value=0, value=1)
fraud_alerts = st.number_input("Fraud Alerts Triggered", min_value=0, value=0)
review_count = st.number_input("Review Count Submitted", min_value=0, value=2)

# Model selection
model_choice = st.selectbox("Choose the Model", ["Linear Regression", "Gradient Boosting", "SVR Linear"])

# Prediction button
if st.button("Predict LTV"):
    input_data = np.array([[
        hash(customer_id) % 100000,  # Encode Customer ID
        age,
        hash(location) % 1000,  # Encode Location
        {"Low": 0, "Medium": 1, "High": 2}[income_level],
        total_transactions,
        avg_transaction_value,
        max_transaction_value,
        min_transaction_value,
        total_spent,
        active_days,
        last_transaction_days_ago,
        loyalty_points_earned,
        referral_count,
        cashback_received,
        app_usage_frequency,
        {"Credit Card": 0, "Debit Card": 1, "Net Banking": 2, "UPI": 3, "Wallet": 4}[preferred_payment_method],
        support_tickets_raised,
        issue_resolution_time,
        customer_satisfaction_score,
        avg_session_duration,
        times_logged_in_last_month,
        promotions_availed,
        returns_made,
        fraud_alerts,
        review_count
    ]])

    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_data)[0]
    elif model_choice == "Gradient Boosting":
        prediction = gbr_model.predict(input_data)[0]
    else:
        prediction = svr_lin_model.predict(input_data)[0]

    st.success(f"Predicted Customer LTV: ₹{prediction:,.2f}")
