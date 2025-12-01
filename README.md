# Fraud Detection with Autoencoder

A student-level but industry-relevant **Fraud Detection System** using an **Autoencoder** for anomaly detection in transaction data. Demonstrates skills in ML, deep learning, feature engineering, and deployment with Flask.

---

## Project Overview

- Detects anomalous transactions in digital payment datasets.
- Uses an **unsupervised autoencoder** trained on normal transactions.
- Flags transactions deviating from learned patterns using **reconstruction error**.
- End-to-end workflow: data generation → preprocessing → model training → evaluation → API deployment.

---

## Repository Structure
fraud-detection-autoencoder/
│
├── src/ # Source code
│ ├── data_prep.py # Synthetic data generation & preprocessing
│ └── model_train.py # Autoencoder model, training, evaluation
│
├── notebooks/ # Notebooks demonstrating workflow
│ └── 01_preprocessing.ipynb
│
├── models/ # Saved model artifacts (autoencoder, scaler info)
│ ├── autoencoder_model.keras
│ └── artifacts.pkl
│
├── app.py # Flask API for real-time predictions
├── requirements.txt # Dependencies
└── README.md # Project overview



---

## Key Features

- Synthetic dataset simulating normal and fraudulent transactions.
- Feature engineering including velocity features (transaction counts, avg amounts).
- Autoencoder learns normal patterns; anomalies detected via reconstruction error.
- Evaluation metrics: **AUC-ROC, Recall, Precision**.
- Flask API for real-time transaction anomaly detection.

---

## Sample Results

| Metric        | Value    |
|---------------|---------|
| AUC-ROC       | 0.9999  |
| Recall        | 1.0     |
| Precision     | 0.1656  |

> High recall ensures most fraudulent transactions are caught; precision is lower due to class imbalance.

---

## Setup & Usage

1. Clone the repository:
git https://github.com/AlgoDove/Fraud-Detection-System.git
cd Fraud-Detection-System

2. Install dependencies:
pip install -r requirements.txt

3. Run the notebooks to explore preprocessing and model training.

4. Start the Flask API:
python app.py

Endpoint: POST /predict
Sample request:
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

Sample response:
[
  {
    "amount": 1200,
    "user_24h_count": 2,
    "user_1h_count": 1,
    "user_amt_diff_24h": 100,
    "merchant_id": "M5",
    "location": "B",
    "reconstruction_error": 198.05,
    "is_anomaly": 1
  }
]


Skills Demonstrated :
Python, NumPy, Pandas for data manipulation
Feature engineering for time-based transaction patterns
TensorFlow/Keras Autoencoder
Model evaluation & visualization
Flask API for deployment

Notes :
Dataset is synthetic but realistic for demonstration.
Model only learns normal transactions; anomalies detected via reconstruction error.
Structured for portfolio and internship relevance.