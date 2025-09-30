from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from config import RANDOM_SEED, NUM_FAULT_CLASSES, FAULT_TYPES

def train_edge_rf_model(X_train, y_train, X_test, y_test):
    """Train lightweight Random Forest model for edge device"""
    print("\nTraining Random Forest model for Edge Device..."); print("-" * 50)
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=5, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print(f"Edge RF Model Performance:\n  - Accuracy: {accuracy_score(y_test, y_pred):.4f}\n  - Macro F1-Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\nClassification Report (Edge Model):"); print(classification_report(y_test, y_pred, target_names=list(FAULT_TYPES.values()), zero_division=0))
    return rf, confusion_matrix(y_test, y_pred)

def train_cloud_dense_model(X_train, y_train, X_test, y_test):
    """Train Dense Neural Network for cloud processing"""
    print("\nTraining Dense Neural Network for Cloud..."); print("-" * 50)
    y_train_cat = to_categorical(y_train, num_classes=NUM_FAULT_CLASSES); y_test_cat = to_categorical(y_test, num_classes=NUM_FAULT_CLASSES)
    
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)), Dropout(0.3),
        Dense(64, activation='relu'), Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(NUM_FAULT_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train_cat, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)
    
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    print(f"Cloud Dense Model Performance:\n  - Accuracy: {accuracy:.4f}\n  - Macro F1-Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\nClassification Report (Cloud Model):"); print(classification_report(y_test, y_pred, target_names=list(FAULT_TYPES.values()), zero_division=0))
    return model, confusion_matrix(y_test, y_pred)