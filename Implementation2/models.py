import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import time
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, Bidirectional, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠ imbalanced-learn not available. Install for better accuracy: pip install imbalanced-learn")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠ LightGBM not available. Please install: pip install lightgbm")

from config import RANDOM_SEED, NUM_FAULT_CLASSES, FAULT_TYPES


def train_edge_lightgbm_model(X_train, y_train, X_test, y_test):
    print("\nTraining Enhanced LightGBM model for Edge Device...")
    print("-" * 50)
    
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is required but not installed. Run: pip install lightgbm")
    
    if SMOTE_AVAILABLE:
        print("Applying FULL SMOTE oversampling for maximum F1-score...")
        smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=3, sampling_strategy='auto')
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"  Original: {X_train.shape[0]} samples")
        print(f"  After SMOTE: {X_train_resampled.shape[0]} samples (fully balanced!)")
        unique, counts = np.unique(y_train_resampled, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"    Class {cls}: {cnt} samples")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
        print("⚠ SMOTE not available, using class weights only")
    
    sample_weights = compute_sample_weight('balanced', y_train_resampled)
    
    lgbm = LGBMClassifier(
        n_estimators=300, 
        max_depth=10,  
        num_leaves=100, 
        learning_rate=0.03,  
        min_child_samples=3,  
        subsample=0.9, 
        colsample_bytree=0.9,  
        reg_alpha=0.05, 
        reg_lambda=0.05, 
        random_state=RANDOM_SEED,
        n_jobs=-1,
        device='cpu',
        verbose=-1,
        importance_type='gain',
        boosting_type='gbdt',
        objective='multiclass',
        num_class=NUM_FAULT_CLASSES
    )
    
    start_time = time.time()
    lgbm.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights)
    training_time = time.time() - start_time
    
    y_pred = lgbm.predict(X_test)
    
    start_inference = time.time()
    for _ in range(100):
        _ = lgbm.predict(X_test[0:1])
    avg_inference_time = (time.time() - start_inference) / 100 * 1000  # ms
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"Edge LightGBM Model Performance:")
    print(f"  - Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  - Macro F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print(f"  - Weighted F1-Score: {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
    print(f"  - Macro Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  - Macro Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"  - Training Time: {training_time:.2f}s")
    print(f"  - Avg Inference Time: {avg_inference_time:.4f}ms")
    
    if f1 >= 0.70:
        print(f"  ✓ TARGET MET: F1-Score >= 70% ✓")
    else:
        print(f"  ⚠ Target not met: F1 = {f1*100:.2f}% (target: 70%)")
    
    print("\nClassification Report (Edge LightGBM Model):")
    print(classification_report(y_test, y_pred, target_names=list(FAULT_TYPES.values()), zero_division=0))
    
    return lgbm, confusion_matrix(y_test, y_pred)


def train_edge_rf_model(X_train, y_train, X_test, y_test):
    print("\nTraining Random Forest model for Edge Device...")
    print("-" * 50)
    
    sample_weights = compute_sample_weight('balanced', y_train)
    
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_split=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    start_time = time.time()
    rf.fit(X_train, y_train, sample_weight=sample_weights)
    training_time = time.time() - start_time
    
    y_pred = rf.predict(X_test)
    
    start_inference = time.time()
    for _ in range(100):
        _ = rf.predict(X_test[0:1])
    avg_inference_time = (time.time() - start_inference) / 100 * 1000
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"Edge RF Model Performance:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - Macro F1-Score: {f1:.4f}")
    print(f"  - Training Time: {training_time:.2f}s")
    print(f"  - Avg Inference Time: {avg_inference_time:.4f}ms")
    
    print("\nClassification Report (Edge RF Model):")
    print(classification_report(y_test, y_pred, target_names=list(FAULT_TYPES.values()), zero_division=0))
    
    return rf, confusion_matrix(y_test, y_pred)


def train_cloud_lstm_model(X_train, y_train, X_test, y_test):
    print("\nTraining Enhanced LSTM Neural Network for Cloud...")
    print("-" * 50)
    
    if SMOTE_AVAILABLE:
        print("Applying optimized SMOTE oversampling (balanced for speed)...")
        from collections import Counter
        class_counts = Counter(y_train)
        majority_count = max(class_counts.values())
        target_count = int(majority_count * 0.3)  
        
        sampling_strategy = {cls: max(count, target_count) 
                            for cls, count in class_counts.items() if count < target_count}
        
        smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=3, sampling_strategy=sampling_strategy)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"  Original: {X_train.shape[0]} samples → After SMOTE: {X_train_resampled.shape[0]} samples")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    X_train_reshaped = X_train_resampled.reshape((X_train_resampled.shape[0], 1, X_train_resampled.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    y_train_cat = to_categorical(y_train_resampled, num_classes=NUM_FAULT_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=NUM_FAULT_CLASSES)
    
    sample_weights = compute_sample_weight('balanced', y_train_resampled)
    
    model = Sequential([
        LSTM(128, input_shape=(1, X_train.shape[1]), return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(NUM_FAULT_CLASSES, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.0001, verbose=0)
    
    start_time = time.time()
    model.fit(
        X_train_reshaped, y_train_cat,
        epochs=50, 
        batch_size=128, 
        validation_split=0.2,
        sample_weight=sample_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    training_time = time.time() - start_time
    
    loss, accuracy = model.evaluate(X_test_reshaped, y_test_cat, verbose=0)
    y_pred = np.argmax(model.predict(X_test_reshaped, verbose=0), axis=1)
    
    start_inference = time.time()
    for _ in range(100):
        _ = model.predict(X_test_reshaped[0:1], verbose=0)
    avg_inference_time = (time.time() - start_inference) / 100 * 1000
    
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"Cloud LSTM Model Performance:")
    print(f"  - Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  - Macro F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print(f"  - Weighted F1-Score: {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
    print(f"  - Macro Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  - Macro Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"  - Training Time: {training_time:.2f}s")
    print(f"  - Avg Inference Time: {avg_inference_time:.4f}ms")
    
    if accuracy >= 0.85:
        print(f"  ✓ TARGET MET: Accuracy >= 85% ✓")
    else:
        print(f"  ⚠ Target not met: Accuracy = {accuracy*100:.2f}% (target: 85%)")
        print(f"     Need {(0.85-accuracy)*100:.2f}% more. Still usable for simulation!")
    
    print("\nClassification Report (Cloud LSTM Model):")
    print(classification_report(y_test, y_pred, target_names=list(FAULT_TYPES.values()), zero_division=0))
    
    return model, confusion_matrix(y_test, y_pred)


def train_cloud_dense_model(X_train, y_train, X_test, y_test):
    print("\nTraining Dense Neural Network for Cloud...")
    print("-" * 50)
    
    y_train_cat = to_categorical(y_train, num_classes=NUM_FAULT_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=NUM_FAULT_CLASSES)
    
    sample_weights = compute_sample_weight('balanced', y_train)
    
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(NUM_FAULT_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    
    start_time = time.time()
    model.fit(
        X_train, y_train_cat,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        sample_weight=sample_weights,
        callbacks=[early_stop],
        verbose=0
    )
    training_time = time.time() - start_time
    
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    start_inference = time.time()
    for _ in range(100):
        _ = model.predict(X_test[0:1], verbose=0)
    avg_inference_time = (time.time() - start_inference) / 100 * 1000
    
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"Cloud Dense Model Performance:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - Macro F1-Score: {f1:.4f}")
    print(f"  - Training Time: {training_time:.2f}s")
    print(f"  - Avg Inference Time: {avg_inference_time:.4f}ms")
    
    print("\nClassification Report (Cloud Dense Model):")
    print(classification_report(y_test, y_pred, target_names=list(FAULT_TYPES.values()), zero_division=0))
    
    return model, confusion_matrix(y_test, y_pred)


def train_edge_svm_model(X_train, y_train, X_test, y_test):
    print("\nTraining SVM model for Edge Device...")
    print("-" * 50)
    
    sample_weights = compute_sample_weight('balanced', y_train)
    
    svm = SVC(kernel='rbf', random_state=RANDOM_SEED, probability=True)
    
    start_time = time.time()
    svm.fit(X_train, y_train, sample_weight=sample_weights)
    training_time = time.time() - start_time
    
    y_pred = svm.predict(X_test)
    
    start_inference = time.time()
    for _ in range(100):
        _ = svm.predict(X_test[0:1])
    avg_inference_time = (time.time() - start_inference) / 100 * 1000
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"Edge SVM Model Performance:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - Macro F1-Score: {f1:.4f}")
    print(f"  - Training Time: {training_time:.2f}s")
    print(f"  - Avg Inference Time: {avg_inference_time:.4f}ms")
    
    print("\nClassification Report (Edge SVM Model):")
    print(classification_report(y_test, y_pred, target_names=list(FAULT_TYPES.values()), zero_division=0))
    
    return svm, confusion_matrix(y_test, y_pred)


def train_cloud_cnn_model(X_train, y_train, X_test, y_test):
    print("\nTraining CNN model for Cloud...")
    print("-" * 50)
    
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    y_train_cat = to_categorical(y_train, num_classes=NUM_FAULT_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=NUM_FAULT_CLASSES)
    
    sample_weights = compute_sample_weight('balanced', y_train)
    
    model = Sequential([
        Conv1D(64, 2, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(2),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(NUM_FAULT_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    
    start_time = time.time()
    model.fit(
        X_train_reshaped, y_train_cat,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        sample_weight=sample_weights,
        callbacks=[early_stop],
        verbose=0
    )
    training_time = time.time() - start_time
    
    loss, accuracy = model.evaluate(X_test_reshaped, y_test_cat, verbose=0)
    y_pred = np.argmax(model.predict(X_test_reshaped, verbose=0), axis=1)
    
    start_inference = time.time()
    for _ in range(100):
        _ = model.predict(X_test_reshaped[0:1], verbose=0)
    avg_inference_time = (time.time() - start_inference) / 100 * 1000
    
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"Cloud CNN Model Performance:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - Macro F1-Score: {f1:.4f}")
    print(f"  - Training Time: {training_time:.2f}s")
    print(f"  - Avg Inference Time: {avg_inference_time:.4f}ms")
    
    print("\nClassification Report (Cloud CNN Model):")
    print(classification_report(y_test, y_pred, target_names=list(FAULT_TYPES.values()), zero_division=0))
    
    return model, confusion_matrix(y_test, y_pred)