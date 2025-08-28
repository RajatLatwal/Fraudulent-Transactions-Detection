import streamlit as st
import numpy as np
import joblib

# --- App Configuration ---
st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="centered")

# --- Custom Styling ---
st.markdown("""
    <style>
    /* Center align title */
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #2E86C1;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555555;
    }
    /* Input box styling */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px;
    }
    /* Button styling */
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-size: 18px;
        font-weight: 600;
        background-color: #2E86C1;
        color: white;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #1A5276;
        color: #f1f1f1;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model ---
try:
    model = joblib.load("fraud_model.pkl")
except FileNotFoundError:
    st.error("⚠️ Model file not found. Please ensure 'fraud_model.pkl' is in the directory.")
    st.stop()

# --- App Header ---
st.markdown('<p class="title">💳 Fraudulent Transaction Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">This app predicts whether a transaction is <b>Fraudulent</b> or <b>Legitimate</b> based on transaction details.</p>', unsafe_allow_html=True)
st.markdown("---")

# --- Input Section ---
st.subheader("📌 Enter Transaction Details")

transaction_types = ['CASH_OUT', 'PAYMENT', 'TRANSFER', 'DEBIT']

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input("⏱️ Time Step (in hours)", min_value=1, max_value=744, value=10)
        amount = st.number_input("💰 Transaction Amount", min_value=0.0, format="%.2f")
        transaction_type = st.selectbox("🔄 Transaction Type", options=transaction_types)

    with col2:
        oldbalanceOrg = st.number_input("🏦 Sender Old Balance", min_value=0.0, format="%.2f")
        newbalanceOrg = st.number_input("🏦 Sender New Balance", min_value=0.0, format="%.2f")
        oldbalanceDest = st.number_input("🎯 Receiver Old Balance", min_value=0.0, format="%.2f")
        newbalanceDest = st.number_input("🎯 Receiver New Balance", min_value=0.0, format="%.2f")

# --- Prediction Section ---
st.markdown("### 🔍 Prediction Result")

if st.button("🚀 Predict Transaction Status"):
    try:
        # Derived features
        diff_orig = oldbalanceOrg - newbalanceOrg
        diff_dest = oldbalanceDest - newbalanceDest
        balance_ratio_orig = newbalanceOrg / (oldbalanceOrg + 1)
        balance_ratio_dest = newbalanceDest / (oldbalanceDest + 1)
        is_large_transfer = 1 if amount > 200000 else 0
        is_zero_balance_origin = 1 if oldbalanceOrg == 0 else 0

        # One-hot encoding
        type_CASH_OUT = 1 if transaction_type == 'CASH_OUT' else 0
        type_DEBIT = 1 if transaction_type == 'DEBIT' else 0
        type_PAYMENT = 1 if transaction_type == 'PAYMENT' else 0
        type_TRANSFER = 1 if transaction_type == 'TRANSFER' else 0

        # Log features
        amount_log = np.log1p(amount)
        balance_ratio_orig_log = np.log1p(balance_ratio_orig)
        balance_ratio_dest_log = np.log1p(balance_ratio_dest)

        # Final feature array
        features = np.array([[step, amount, oldbalanceOrg, newbalanceOrg,
                              oldbalanceDest, newbalanceDest,
                              diff_orig, diff_dest,
                              balance_ratio_orig, balance_ratio_dest,
                              is_large_transfer, is_zero_balance_origin,
                              type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER,
                              amount_log, balance_ratio_orig_log, balance_ratio_dest_log]])

        # Predict
        prediction = model.predict(features)
        probability = model.predict_proba(features)

        if prediction[0] == 1:
            fraud_prob = probability[0][1] * 100
            st.error(f"⚠️ Fraudulent Transaction Detected!\n\nConfidence: **{fraud_prob:.2f}%**")
            st.snow()
        else:
            legit_prob = probability[0][0] * 100
            st.success(f"✅ Legitimate Transaction\n\nConfidence: **{legit_prob:.2f}%**")
            st.balloons()

    except Exception as e:
        st.error(f"Prediction error: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Built with ❤️ using Streamlit & Machine Learning</p>", unsafe_allow_html=True)
