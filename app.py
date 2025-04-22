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

# Input fields (replace these with your real features)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
total_transactions = st.number_input("Total Transactions", min_value=0, value=5)
avg_transaction_value = st.number_input("Average Transaction Value", min_value=0.0, value=100.0)
total_spent = st.number_input("Total Spent", min_value=0.0, value=500.0)

# Add more fields as per your real dataset...

# Model selection
model_choice = st.selectbox("Choose the Model", ["Linear Regression", "Gradient Boosting", "SVR Linear"])

# Prediction button
if st.button("Predict LTV"):
    input_data = np.array([[age, total_transactions, avg_transaction_value, total_spent]])  # Adjust dimensions!
    
    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_data)[0]
    elif model_choice == "Gradient Boosting":
        prediction = gbr_model.predict(input_data)[0]
    else:
        prediction = svr_lin_model.predict(input_data)[0]

    st.success(f"Predicted Customer LTV: ₹{prediction:,.2f}")
