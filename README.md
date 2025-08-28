# 💳 Fraudulent Transaction Detection App

**Fraud Detection App** is a web application built using **Streamlit** that predicts whether a financial transaction is **Fraudulent** or **Legitimate**.  
It uses a trained **RandomForestClassifier** along with feature engineering techniques to deliver fast and accurate predictions.

---

## 🔗 Live Demo

👉 Try it now:  
**[Fraudulent Transaction Detection App](https://fraudulent-transactions-detection.streamlit.app/)**

---

## 🎯 Features

- 🔍 Detects fraudulent transactions in real-time  
- 🧠 Powered by a trained **Machine Learning model**  
- ⚙️ Automatic feature engineering from inputs:
  - Balance differences & ratios  
  - Log transformations  
  - One-hot encoding of transaction type  
  - Flags for large transfers & zero balance accounts  
- 📊 Provides **confidence scores (%)** for transparency  
- 📱 Clean and responsive **Streamlit UI**  

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – interactive web app
- **Scikit-learn** – RandomForestClassifier model
- **Pandas & NumPy** – data preprocessing & feature engineering
- **Joblib** – for model serialization

---

## 🚀 How It Works

1. Enter transaction details:
   - Time Step (1–744 hours)  
   - Transaction Amount  
   - Sender’s & Receiver’s balances  
   - Transaction Type (CASH_OUT, PAYMENT, TRANSFER, DEBIT)  
2. Click **Predict**  
3. Get instant results with a **Fraudulent / Legitimate** label and confidence score  

---
# Install dependencies
pip install -r requirements.txt
