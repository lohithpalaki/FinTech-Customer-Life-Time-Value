
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Title
st.title("Customer Lifetime Value (LTV) Predictor")
st.write("Fill in the details to predict customer LTV")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
total_transactions = st.number_input("Total Transactions", min_value=0, value=10)
avg_transaction_value = st.number_input("Average Transaction Value", min_value=0.0, value=150.0)
app_usage_frequency = st.selectbox("App Usage Frequency", ["Daily", "Weekly", "Monthly"])
customer_satisfaction_score = st.slider("Customer Satisfaction Score", 0.0, 10.0, 7.5)

# Model selection
model_choice = st.selectbox("Choose the Model", ["Linear Regression", "Gradient Boosting", "SVR"])

# Mapping categorical variable
usage_encoded = [
    1 if app_usage_frequency == "Daily" else 0,
    1 if app_usage_frequency == "Weekly" else 0,
    1 if app_usage_frequency == "Monthly" else 0
]

# Prediction
if st.button("Predict LTV"):
    input_array = np.array([[
        age, total_transactions, avg_transaction_value,
        customer_satisfaction_score
    ] + usage_encoded])

    model_file = {
        "Linear Regression": "lr_model.pkl",
        "Gradient Boosting": "gbr_model.pkl",
        "SVR": "svr_lin_model.pkl"
    }[model_choice]

    model = joblib.load(model_file)
    prediction = model.predict(input_array)[0]
    st.success(f"Predicted Customer LTV: ₹{prediction:,.2f}")
