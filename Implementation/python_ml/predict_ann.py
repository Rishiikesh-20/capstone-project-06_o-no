import os
import sys
import json
import numpy as np
import ast
import time
import joblib
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def extract_vibration_features(vibration_series):
    """Extract statistical features from vibration time series"""
    if isinstance(vibration_series, list):
        data = np.array(vibration_series, dtype=float)
    else:
        data = np.array(vibration_series, dtype=float)
    
    if np.all(np.isnan(data)):
        return [0.0, 0.0, 0.0, 0.0, 0.0]  
    
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    features = [
        np.mean(clean_data),      
        np.std(clean_data),       
        np.max(clean_data),       
        np.min(clean_data),       
        len(clean_data) / len(data) 
    ]
    
    return features

def predict_fault(input_file):
    """Predict fault using ANN model"""
    start_time = time.time()
    
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "ann_model.keras")
    scaler_path = os.path.join(script_dir, "ann_scaler.joblib")
    imputer_path = os.path.join(script_dir, "ann_imputer.joblib")
    config_path = os.path.join(script_dir, "ann_config.json")
    
    try:
        # Load model and preprocessing objects
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ANN model not found: {model_path}")
        
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        imputer = joblib.load(imputer_path)
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        threshold = config.get('threshold', 0.5)
        
        # Load input data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Extract features
        vibration = data.get('vibration', [0] * 100)
        temp = data.get('temp', 50.0)
        voltage = data.get('voltage', 220.0)
        
        # Extract vibration features
        vib_features = extract_vibration_features(vibration)
        
        # Create feature vector: [temp, voltage, vib_mean, vib_std, vib_max, vib_min, vib_completeness]
        features = np.array([[temp, voltage] + vib_features], dtype=np.float32)
        
        # Preprocessing
        features_imputed = imputer.transform(features)
        features_scaled = scaler.transform(features_imputed)
        
        # Prediction
        prob = float(model.predict(features_scaled, verbose=0)[0][0])
        fault = 1 if prob > threshold else 0
        
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return {
            "fault": fault,
            "prob": prob,
            "latency_ms": latency,
            "method": "ann",
            "threshold": threshold,
            "temp": temp,
            "voltage": voltage,
            "features_used": len(vib_features) + 2  # vibration features + temp + voltage
        }
        
    except Exception as e:
        # Fallback to simple threshold-based prediction
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
        except:
            data = {}
        
        temp = data.get('temp', 50.0)
        voltage = data.get('voltage', 220.0)
        
        # Simple rule-based fallback
        fault = 1 if (temp > 65 or voltage > 250 or voltage < 200) else 0
        prob = 0.8 if fault == 1 else 0.2
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "fault": fault,
            "prob": prob,
            "latency_ms": latency,
            "method": "fallback_threshold",
            "error": str(e),
            "temp": temp,
            "voltage": voltage
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python predict_ann.py <input_file>"}))
        sys.exit(1)
    
    input_file = sys.argv[1]
    result = predict_fault(input_file)
    print(json.dumps(result))
