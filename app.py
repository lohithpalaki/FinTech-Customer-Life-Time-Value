import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model (ensure it's saved without any scaler applied)
model = joblib.load("rfc_model.pkl")

# Feature columns used in the model
feature_cols = [
    'Age', 'Account_Balance', 'Transaction_Amount',
    'Account_Balance_After_Transaction', 'Loan_Amount', 'Interest_Rate',
    'Loan_Term', 'Credit_Limit', 'Credit_Card_Balance',
    'Minimum_Payment_Due', 'Rewards_Points'
]

# Define manual prediction input page
def single_prediction():
    st.title("📊 Banking Risk Prediction - Manual Entry")

    input_data = []
    for col in feature_cols:
        value = st.number_input(f"Enter {col.replace('_', ' ')}:", value=0.0)
        input_data.append(value)

    if st.button("Predict Risk"):
        data = np.array([input_data])
        prediction = model.predict(data)[0]
        st.success(f"🧠 Predicted Risk Category: **{prediction}**")

# Define batch prediction input page
def batch_prediction():
    st.title("📂 Banking Risk Prediction - Batch from CSV")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if all(col in df.columns for col in feature_cols):
            preds = model.predict(df[feature_cols])
            df["Predicted_Risk"] = preds
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", csv, "predicted_risk.csv", "text/csv")
        else:
            st.error(f"CSV must contain the following columns: {', '.join(feature_cols)}")

# Simple login (optional)
auth_users = {"RiskPredict": "Secure123"}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in auth_users and auth_users[username] == password:
            st.session_state.logged_in = True
            st.success("✅ Login successful!")
        else:
            st.error("❌ Invalid credentials")

# Main app logic
def main():
    if not st.session_state.logged_in:
        login()
    else:
        st.sidebar.title("Navigation")
        option = st.sidebar.radio("Choose a page", ["Single Prediction", "Batch Prediction"])
        if option == "Single Prediction":
            single_prediction()
        elif option == "Batch Prediction":
            batch_prediction()

if __name__ == "__main__":
    main()
