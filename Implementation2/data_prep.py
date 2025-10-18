import numpy as np
import pandas as pd
import warnings

from config import FEATURE_COLUMNS, FAULT_TYPES, DATASET_PATH

def load_and_prepare_dataset():
    """Load your specific dataset and prepare it for the models with enhanced error handling."""
    try:
        print(f"Attempting to load dataset from {DATASET_PATH}...")
        df = pd.read_csv(DATASET_PATH)
        
        if df.empty:
            raise ValueError(f"Dataset at {DATASET_PATH} is empty!")
        
        print(f"✓ Successfully loaded {len(df)} samples from {DATASET_PATH}")

        required_cols = FEATURE_COLUMNS + ['TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Machine failure']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        conditions = [
            (df['TWF'] == 1),
            (df['HDF'] == 1),
            (df['PWF'] == 1),
            (df['OSF'] == 1),
            (df['RNF'] == 1),
            (df['Machine failure'] == 1) 
        ]
        choices = [1, 2, 3, 4, 5, 5]
        df['Fault_Label'] = np.select(conditions, choices, default=0)
        
        normal_ratio = (df['Fault_Label'] == 0).sum() / len(df)
        if normal_ratio > 0.7:
            warnings.warn(f"⚠ Dataset is highly imbalanced! Normal class: {normal_ratio*100:.1f}%. "
                         f"Using class weights to mitigate.", UserWarning)
        
        final_cols = FEATURE_COLUMNS + ['Fault_Label']
        df_final = df[final_cols]
        
        if df_final.isnull().any().any():
            warnings.warn("Dataset contains missing values. Filling with column means.", UserWarning)
            df_final = df_final.fillna(df_final.mean())
        
        print("✓ Dataset preprocessed: Created 'Fault_Label' and selected feature columns.")
        return df_final
            
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{DATASET_PATH}'. Please check the file path.")
        exit()
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        exit()

def generate_runtime_sensor_data(scenario='Normal'):
    """Generate single sensor reading for runtime simulation, matching your dataset's features."""
    sample = {}
    fault_map = {v: k for k, v in FAULT_TYPES.items()}
    
    sample.update({
        'Air temperature [K]': np.random.normal(300, 2),
        'Process temperature [K]': np.random.normal(310, 2),
        'Rotational speed [rpm]': np.random.normal(1500, 150),
        'Torque [Nm]': np.random.normal(40, 10),
        'Tool wear [min]': np.random.normal(100, 50)
    })

    if scenario == 'Tool Wear Failure':
        sample['Torque [Nm]'] *= 1.5
        sample['Tool wear [min]'] = np.random.normal(220, 20) 
    elif scenario == 'Heat Dissipation Failure':
        sample['Air temperature [K]'] += np.random.uniform(5, 10)
        sample['Process temperature [K]'] += np.random.uniform(5, 15)
    elif scenario == 'Power Failure':
        sample['Torque [Nm]'] = np.random.normal(70, 10)
        sample['Rotational speed [rpm]'] -= np.random.uniform(200, 400) 
    elif scenario == 'Overstrain Failure':
        sample['Torque [Nm]'] = np.random.normal(75, 10) 
        sample['Tool wear [min]'] += np.random.uniform(20, 40)
        
    sample['Tool wear [min]'] = max(0, sample['Tool wear [min]'])
    sample['Fault_Label'] = fault_map[scenario]
    return pd.DataFrame([sample])