import streamlit as st
import joblib
import numpy as np

st.title("Simple LTV Predictor")

total_spent = st.number_input("Total Spent", 0.0, 100000.0, 5000.0)
loyalty_points_earned = st.number_input("Loyalty Points Earned", 0, 10000, 250)
referral_count = st.number_input("Referral Count", 0, 100, 5)
cashback_received = st.number_input("Cashback Received", 0.0, 10000.0, 300.0)
customer_satisfaction_score = st.slider("Customer Satisfaction Score", 0.0, 10.0, 7.5)

if st.button("Predict"):
    model = joblib.load("lr_model.pkl")
    input_data = np.array([[
        total_spent,
        loyalty_points_earned,
        referral_count,
        cashback_received,
        customer_satisfaction_score
    ]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted LTV: ₹{prediction:,.2f}")
