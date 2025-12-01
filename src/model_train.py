import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import pickle
import os

def create_autoencoder(input_dim):
    """
    Defines the deep autoencoder architecture (Encoder -> Bottleneck -> Decoder).

    Args:
        input_dim (int): The number of features (columns) in the input data.

    Returns:
        tensorflow.keras.Model: The compiled Autoencoder model.
    """
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoder = Dense(64, activation="relu")(input_layer)
    encoder = Dense(32, activation="relu")(encoder)
    
    # Bottleneck (Compression) - Forces model to learn essential normal patterns
    bottleneck = Dense(8, activation="relu", name="bottleneck")(encoder)
    
    # Decoder
    decoder = Dense(32, activation="relu")(bottleneck)
    decoder = Dense(64, activation="relu")(decoder)
    
    # Output layer (reconstruction)
    output_layer = Dense(input_dim, activation="linear")(decoder)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    
    return model

def calculate_reconstruction_error(model, X):
    """Calculates the Mean Squared Error (MSE) between input and reconstruction."""
    predictions = model.predict(X, verbose=0)
    # MSE is calculated across the features (axis=1) for each sample
    mse = np.mean(np.power(X - predictions, 2), axis=1)
    return mse

def find_optimal_threshold(X_val, model):
    """
    Determines the operational anomaly threshold using the 95th percentile 
    of the reconstruction error on the normal validation set.
    """
    mse_val = calculate_reconstruction_error(model, X_val)
    
    # Sets the threshold where 5% of normal transactions will be flagged as False Positives.
    threshold = np.percentile(mse_val, 95) 
    
    return threshold

def train_and_evaluate(X_train, X_val, X_test, y_test, artifacts, epochs=20):
    """
    Trains the autoencoder, evaluates, and saves the model + artifacts.
    Uses artifacts['feature_columns'] already saved from preprocessing.
    """
    input_dim = X_train.shape[1]
    model = create_autoencoder(input_dim)
    
    print(f"Training Autoencoder with {input_dim} features for {epochs} epochs...")
    
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=128,
        validation_data=(X_val, X_val),
        shuffle=True,
        verbose=1
    )
    
    threshold = find_optimal_threshold(X_val, model)
    print(f"\nOptimal Anomaly Threshold (95th percentile): {threshold:.4f}")
    
    mse_test = calculate_reconstruction_error(model, X_test)
    y_pred = (mse_test >= threshold).astype(int)
    
    # Metrics
    auc_roc = roc_auc_score(y_test, mse_test) 
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[1, 1]; FP = cm[0, 1]; FN = cm[1, 0]
    recall_at_threshold = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision_at_threshold = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    print("\n--- Model Performance on Test Set ---")
    print(f"AUC-ROC Score: {auc_roc:.4f}")
    print(f"Recall (Fraud Catch Rate): {recall_at_threshold:.4f}")
    print(f"Precision (Alert Accuracy): {precision_at_threshold:.4f}")
    
    # Save artifacts
    artifacts['threshold'] = threshold
    # ✅ DO NOT touch feature_columns here — already saved in preprocess_data

    import os
    import pickle
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_dir = os.path.join(project_root, 'models')
    os.makedirs(save_dir, exist_ok=True)

    model.save(os.path.join(save_dir, 'autoencoder_model.keras'))
    with open(os.path.join(save_dir, 'artifacts.pkl'), 'wb') as f:
        pickle.dump(artifacts, f)
        
    print(f"\nModel and artifacts saved successfully to '{save_dir}'")
    
    return history, model



