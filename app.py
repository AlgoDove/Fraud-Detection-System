from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import os

# -----------------------------
# CONFIG / PATHS
# -----------------------------
MODEL_PATH = os.path.join("models", "autoencoder_model.keras")
ARTIFACTS_PATH = os.path.join("models", "artifacts.pkl")

# -----------------------------
# LOAD MODEL AND ARTIFACTS
# -----------------------------
autoencoder = tf.keras.models.load_model(MODEL_PATH)
with open(ARTIFACTS_PATH, "rb") as f:
    artifacts = pickle.load(f)

SCALER = artifacts['scaler']
FEATURE_COLUMNS = artifacts['feature_columns']
THRESHOLD = artifacts['threshold']

NUMERIC_FEATURES = artifacts.get('numeric_features', ['amount', 'user_24h_count', 'user_1h_count', 'user_amt_diff_24h'])
CATEGORICAL_FEATURES = artifacts.get('categorical_features', ['merchant_id', 'location'])

# -----------------------------
# FLASK SETUP
# -----------------------------
app = Flask(__name__)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def preprocess_input(df: pd.DataFrame):
    """
    Apply scaling and one-hot encoding to match training features
    """
    df_proc = df.copy()

    # Scale numeric
    df_proc[NUMERIC_FEATURES] = SCALER.transform(df_proc[NUMERIC_FEATURES])

    # One-hot encode categoricals
    df_proc = pd.get_dummies(df_proc, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int)

    # Ensure all columns exist
    df_proc = df_proc.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df_proc.values.astype(np.float32)

def predict_anomalies(df: pd.DataFrame):
    X = preprocess_input(df)
    reconstructions = autoencoder.predict(X, verbose=0)
    mse = np.mean(np.power(X - reconstructions, 2), axis=1)
    df['reconstruction_error'] = mse
    df['is_anomaly'] = (mse >= THRESHOLD).astype(int)
    return df

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return "<h2>Fraud Detection Autoencoder API is running!</h2>"

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON payload with transactions to predict
    Example:
    {
      "transactions": [
        {
          "amount": 1200,
          "user_24h_count": 2,
          "user_1h_count": 1,
          "user_amt_diff_24h": 100,
          "merchant_id": "M5",
          "location": "B"
        }
      ]
    }
    """
    data = request.get_json()
    if not data or 'transactions' not in data:
        return jsonify({"error": "Invalid payload, must include 'transactions'"}), 400

    df_input = pd.DataFrame(data['transactions'])
    df_output = predict_anomalies(df_input)
    return df_output.to_json(orient="records")

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
