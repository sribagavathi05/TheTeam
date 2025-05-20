import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd

# Load the trained model
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Load the trained model weights and scaler
@st.cache_resource
def load_model_and_scaler():
    scaler = joblib.load("scaler.pkl")
    model = FraudDetectionModel(input_dim=29)  # Adjust input_dim based on your feature size
    model.load_state_dict(torch.load("pytorch_fraud_detection_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model, scaler

model, scaler = load_model_and_scaler()

# Streamlit App
st.title("Fraud Detection App")
st.write("Upload a CSV file with transaction data to predict fraud.")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.write(data.head())

    # Drop unnecessary columns if they exist
    cols_to_drop = ['nameOrig', 'nameDest']
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns], errors='ignore')

    # One-hot encode the 'type' column if present
    if 'type' in data.columns:
        data = pd.get_dummies(data, columns=['type'])
    
    # Ensure columns match the trained model
    expected_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                         'oldbalanceDest', 'newbalanceDest', 'type_CASH_IN', 
                         'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 
                         'type_TRANSFER']  # Add as per your dataset
    for col in expected_features:
        if col not in data.columns:
            data[col] = 0  # Add missing columns with default value 0

    data = data[expected_features]

    # Scale the data
    scaled_data = scaler.transform(data)

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32)

    # Make predictions
    with torch.no_grad():
        predictions = model(input_tensor).squeeze().numpy()
        predictions = (predictions > 0.5).astype(int)  # Threshold at 0.5

    # Append predictions to the data
    data['isFraud'] = predictions
    st.write("Predictions:")
    st.write(data)

    # Downloadable CSV
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='fraud_predictions.csv',
        mime='text/csv',
    )
