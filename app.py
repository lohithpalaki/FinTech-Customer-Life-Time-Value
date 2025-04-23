import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Dummy credentials for login
auth_users = {"Customer_LTV": "Fintech"}

# Session state to manage login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Login", "Predict One", "Predict from CSV"])

# Page 1: Login
def login_page():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in auth_users and auth_users[username] == password:
            st.session_state.logged_in = True
            st.success("Login successful! Use the sidebar to navigate.")
        else:
            st.error("Invalid credentials")

# Page 2: Single Prediction
def single_prediction_page():
    if not st.session_state.logged_in:
        st.warning("Please log in to access this page.")
        return

    st.title("Simple LTV Predictor")

    total_spent = st.number_input("Total Spent", 0.0, 10000000.0, 5000.0)
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

# Page 3: Batch Prediction
def batch_prediction_page():
    if not st.session_state.logged_in:
        st.warning("Please log in to access this page.")
        return

    st.title("📂 Predict LTV from CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", df.head())

        required_cols = ['Total_Spent', 'Loyalty_Points_Earned', 'Referral_Count', 'Cashback_Received', 'Customer_Satisfaction_Score']
        if all(col in df.columns for col in required_cols):
            model = joblib.load("lr_model.pkl")
            predictions = model.predict(df[required_cols])
            df['Predicted_LTV'] = predictions
            st.write("Predictions:", df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predicted_ltv.csv", "text/csv")
        else:
            st.error(f"CSV must contain columns: {', '.join(required_cols)}")

# Router
if page == "Login":
    login_page()
elif page == "Predict One":
    single_prediction_page()
elif page == "Predict from CSV":
    batch_prediction_page()
