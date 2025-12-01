import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(n_normal=5000, n_fraud=50):
    np.random.seed(42)
    
    # Normal Transactions
    normal_data = {
        'timestamp': pd.to_datetime(pd.date_range('2023-01-01', periods=n_normal, freq='h').to_numpy() + 
                                    np.random.randint(0, 3600, n_normal) * np.timedelta64(1, 's')),
        'user_id': np.random.randint(1, 500, n_normal),
        'amount': np.random.lognormal(mean=2.5, sigma=0.5, size=n_normal),
        'merchant_id': np.random.choice([f'M{i}' for i in range(1, 20)], n_normal),
        'location': np.random.choice(['A', 'B', 'C', 'D'], n_normal),
        'is_fraud': 0
    }
    df_normal = pd.DataFrame(normal_data)

    # Fraudulent Transactions
    fraud_data = {
        'timestamp': pd.to_datetime(pd.to_datetime('2023-01-20 14:00:00') + 
                                    np.random.randint(0, 600, n_fraud) * np.timedelta64(1, 's')),
        'user_id': np.random.choice([10, 15, 20], n_fraud),
        'amount': np.random.lognormal(mean=5.0, sigma=0.8, size=n_fraud),
        'merchant_id': np.random.choice([f'M{i}' for i in range(15, 20)], n_fraud),
        'location': np.random.choice(['Z'], n_fraud),
        'is_fraud': 1
    }
    df_fraud = pd.DataFrame(fraud_data)

    df = pd.concat([df_normal, df_fraud], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    return df

def engineer_velocity_features(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    df['user_24h_count'] = 0
    df['user_1h_count'] = 0
    df['user_amt_diff_24h'] = 0.0

    for user_id, group in df.groupby('user_id'):
        group = group.sort_values('timestamp')
        temp = group.set_index('timestamp')
        temp['user_24h_count'] = temp['amount'].rolling('24h').count() - 1
        temp['user_1h_count'] = temp['amount'].rolling('1h').count() - 1
        temp['user_24h_avg'] = temp['amount'].rolling('24h').mean()
        temp['user_amt_diff_24h'] = temp['amount'] - temp['user_24h_avg']
        temp = temp.fillna(0)
        df.loc[group.index, 'user_24h_count'] = temp['user_24h_count'].values
        df.loc[group.index, 'user_1h_count'] = temp['user_1h_count'].values
        df.loc[group.index, 'user_amt_diff_24h'] = temp['user_amt_diff_24h'].values

    return df

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df: pd.DataFrame):
    """
    Preprocesses synthetic transaction data for Autoencoder training.
    - Engineers velocity features
    - Scales numeric features
    - One-hot encodes categorical features
    - Splits into train/val/test sets
    - Returns artifacts including scaler and feature columns
    """
    # --- Feature engineering ---
    df_processed = engineer_velocity_features(df)  # keep your existing engineer_velocity_features

    # --- Split normal and fraud ---
    df_normal = df_processed[df_processed['is_fraud'] == 0]
    df_fraud = df_processed[df_processed['is_fraud'] == 1]

    # Training and validation only on normal data
    X_train_normal, X_val_normal = train_test_split(
        df_normal.drop(columns=['timestamp', 'user_id']), test_size=0.15, random_state=42
    )

    # Full test set (normal + fraud)
    X_test_all = pd.concat([df_fraud, df_normal], ignore_index=True)

    # Pop target
    y_test = X_test_all.pop('is_fraud')
    X_test_all.drop(columns=['timestamp', 'user_id'], inplace=True)

    # --- Preprocessing ---
    numeric_features = ['amount', 'user_24h_count', 'user_1h_count', 'user_amt_diff_24h']
    categorical_features = ['merchant_id', 'location']

    # Scale numeric features
    scaler = StandardScaler()
    X_train_normal[numeric_features] = scaler.fit_transform(X_train_normal[numeric_features])

    # One-hot encode categorical features in training
    X_train_encoded = pd.get_dummies(X_train_normal, columns=categorical_features, drop_first=True, dtype=int)
    all_columns = X_train_encoded.columns.tolist()  # save all columns for consistent transformation

    # Helper function to transform validation/test sets
    def transform_set(X_set, scaler, numeric_features, categorical_features, all_columns):
        X_set[numeric_features] = scaler.transform(X_set[numeric_features])
        X_set_encoded = pd.get_dummies(X_set, columns=categorical_features, drop_first=True, dtype=int)
        # Reindex to match training columns, fill missing columns with 0
        return X_set_encoded.reindex(columns=all_columns, fill_value=0)

    X_val_final = transform_set(X_val_normal.copy(), scaler, numeric_features, categorical_features, all_columns)
    X_test_final = transform_set(X_test_all.copy(), scaler, numeric_features, categorical_features, all_columns)

    # Convert to NumPy arrays and force float32 for TensorFlow
    X_train = X_train_encoded.values.astype(np.float32)
    X_val = X_val_final.values.astype(np.float32)
    X_test = X_test_final.values.astype(np.float32)
    y_test = y_test.values.astype(np.float32)

    # Save preprocessing artifacts
    artifacts = {
        'scaler': scaler,
        'feature_columns': all_columns,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }

    return X_train, X_val, X_test, y_test, artifacts

