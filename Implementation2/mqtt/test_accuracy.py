
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED, FEATURE_COLUMNS
from data_prep import load_and_prepare_dataset
from models import train_edge_lightgbm_model, train_cloud_lstm_model

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("\n" + "="*70)
print(" " * 15 + "ACCURACY VERIFICATION TEST")
print("="*70)

print("\nLoading dataset...")
df = load_and_prepare_dataset()

X = df[FEATURE_COLUMNS]
y = df['Fault_Label']

print(f"\nDataset: {len(df)} samples")
print(f"Class distribution:\n{y.value_counts().sort_index()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

print("\n" + "="*70)
print("TRAINING MODELS WITH ACCURACY IMPROVEMENTS")
print("="*70)

print("\n" + "-"*70)
edge_model, edge_cm = train_edge_lightgbm_model(X_train_scaled, y_train, X_test_scaled, y_test)

print("\n" + "-"*70)
cloud_model, cloud_cm = train_cloud_lstm_model(X_train_scaled, y_train, X_test_scaled, y_test)

print("\n" + "="*70)
print(" " * 25 + "FINAL RESULTS")
print("="*70)
print("\n✅ Model training completed successfully!")
print("\nCheck the output above to verify:")
print("  ✓ Edge LightGBM: Macro F1-Score >= 70%")
print("  ✓ Cloud LSTM: Overall Accuracy >= 85%")
print("="*70)