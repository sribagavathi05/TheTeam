import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load models
xgb_model = joblib.load("xgb.pkl")
scaler = joblib.load("scaler.pkl")
autoencoder = tf.keras.models.load_model("autoencoder.h5")

# Define your threshold
THRESHOLD = 0.01  # Update this based on your training

# Define the number of features (adjust as needed)
n_features = 30

st.set_page_config(page_title="Fraud Detection with XGBoost & Autoencoder")
st.title("Fraud Detection App")
st.markdown("Enter transaction details below to predict fraud using both XGBoost and Autoencoder models.")

# Input form
with st.form("fraud_form"):
    features = []
    for i in range(n_features):
        val = st.number_input(f"Feature {i+1}", value=0.0, step=0.01, format="%.5f")
        features.append(val)
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Convert and scale input
        input_array = np.array(features).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # XGBoost prediction
        xgb_pred = xgb_model.predict(scaled_input)[0]

        # Autoencoder anomaly score
        reconstruction = autoencoder.predict(scaled_input)
        mse = np.mean(np.power(scaled_input - reconstruction, 2))
        anomaly = mse > THRESHOLD

        # Show results
        st.success(f"**XGBoost:** {'Fraud' if xgb_pred else 'Legit'}")
        st.info(f"**Autoencoder:** {'Anomaly' if anomaly else 'Normal'} (MSE: {mse:.5f})")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
