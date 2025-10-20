import numpy as np
import pandas as pd
import warnings

from config import FEATURE_COLUMNS, FAULT_TYPES, DATASET_PATH

def augment_minority_classes(df, augmentation_factor=10):
    """
    Augment minority classes using intelligent noise injection to improve class balance.
    This helps models learn better representations of rare fault types.
    """
    augmented_dfs = [df.copy()]
    
    # Calculate class distribution
    class_counts = df['Fault_Label'].value_counts()
    print(f"\n  Original class distribution:")
    for cls in sorted(class_counts.index):
        print(f"    Class {cls} ({FAULT_TYPES[cls]}): {class_counts[cls]} samples")
    
    # Identify minority classes (less than 1% of dataset)
    threshold = len(df) * 0.01
    minority_classes = [cls for cls, count in class_counts.items() if count < threshold and cls != 0]
    
    print(f"\n  Augmenting minority classes: {[FAULT_TYPES[cls] for cls in minority_classes]}")
    
    for fault_class in minority_classes:
        class_data = df[df['Fault_Label'] == fault_class].copy()
        original_count = len(class_data)
        
        if original_count == 0:
            continue
            
        # Create synthetic samples with controlled noise
        for i in range(augmentation_factor):
            augmented_class = class_data.copy()
            
            # Add Gaussian noise to numerical features (preserve patterns)
            for col in FEATURE_COLUMNS:
                if col in augmented_class.columns:
                    # Noise proportional to feature's standard deviation
                    noise_std = augmented_class[col].std() * 0.05  # 5% noise
                    noise = np.random.normal(0, noise_std, size=len(augmented_class))
                    augmented_class[col] = augmented_class[col] + noise
            
            augmented_dfs.append(augmented_class)
        
        print(f"    Class {fault_class}: {original_count} → {original_count * (augmentation_factor + 1)} samples")
    
    # Combine original and augmented data
    result_df = pd.concat(augmented_dfs, ignore_index=True)
    
    # Shuffle to mix augmented samples
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n  Final dataset size: {len(df)} → {len(result_df)} samples")
    print(f"  ✓ Data augmentation complete!\n")
    
    return result_df

def load_and_prepare_dataset(augment_minority=True, augmentation_factor=10):
    """Load your specific dataset and prepare it for the models with enhanced error handling and data augmentation."""
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
            print(f"⚠ Dataset is highly imbalanced! Normal class: {normal_ratio*100:.1f}%.")
            if augment_minority:
                print(f"  → Applying data augmentation to minority classes (factor: {augmentation_factor}x)...")
        
        final_cols = FEATURE_COLUMNS + ['Fault_Label']
        df_final = df[final_cols]
        
        if df_final.isnull().any().any():
            warnings.warn("Dataset contains missing values. Filling with column means.", UserWarning)
            df_final = df_final.fillna(df_final.mean())
        
        # Apply data augmentation for minority classes
        if augment_minority:
            df_final = augment_minority_classes(df_final, augmentation_factor)
        
        print("✓ Dataset preprocessed: Created 'Fault_Label' and selected feature columns.")
        return df_final
            
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{DATASET_PATH}'. Please check the file path.")
        exit()
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        exit()

def generate_runtime_sensor_data(scenario='Normal', use_realistic_patterns=True):
    """
    Generate single sensor reading for runtime simulation, matching your dataset's features.
    Now uses more realistic patterns based on actual training data statistics.
    """
    sample = {}
    fault_map = {v: k for k, v in FAULT_TYPES.items()}
    
    if use_realistic_patterns:
        # More realistic base values matching training data distribution
        # These values are based on typical manufacturing sensor readings
        if scenario == 'Normal':
            sample.update({
                'Air temperature [K]': np.random.normal(300, 2),
                'Process temperature [K]': np.random.normal(310, 1.5),
                'Rotational speed [rpm]': np.random.normal(1500, 100),
                'Torque [Nm]': np.random.normal(40, 8),
                'Tool wear [min]': np.random.normal(100, 40)
            })
        
        elif scenario == 'Tool Wear Failure':
            # Tool wear causes increased torque and high tool wear time
            sample.update({
                'Air temperature [K]': np.random.normal(300, 2),
                'Process temperature [K]': np.random.normal(310, 2),
                'Rotational speed [rpm]': np.random.normal(1500, 100),
                'Torque [Nm]': np.random.normal(60, 12),  # Significantly higher torque
                'Tool wear [min]': np.random.normal(200, 30)  # Very high wear
            })
        
        elif scenario == 'Heat Dissipation Failure':
            # Heat issues: elevated temperatures with normal operations
            sample.update({
                'Air temperature [K]': np.random.normal(310, 3),  # Higher air temp
                'Process temperature [K]': np.random.normal(320, 3),  # Higher process temp
                'Rotational speed [rpm]': np.random.normal(1500, 100),
                'Torque [Nm]': np.random.normal(40, 8),
                'Tool wear [min]': np.random.normal(100, 40)
            })
        
        elif scenario == 'Power Failure':
            # Power issues: high torque with reduced speed
            sample.update({
                'Air temperature [K]': np.random.normal(300, 2),
                'Process temperature [K]': np.random.normal(310, 2),
                'Rotational speed [rpm]': np.random.normal(1200, 150),  # Lower speed
                'Torque [Nm]': np.random.normal(70, 15),  # Very high torque
                'Tool wear [min]': np.random.normal(100, 40)
            })
        
        elif scenario == 'Overstrain Failure':
            # Overstrain: very high torque, increased wear, possible temp rise
            sample.update({
                'Air temperature [K]': np.random.normal(302, 2),
                'Process temperature [K]': np.random.normal(312, 2),
                'Rotational speed [rpm]': np.random.normal(1500, 100),
                'Torque [Nm]': np.random.normal(80, 15),  # Extreme torque
                'Tool wear [min]': np.random.normal(130, 50)  # Elevated wear
            })
        
        elif scenario == 'Random Failure':
            # Random failure: unpredictable combinations
            # Randomly choose between different failure patterns
            failure_type = np.random.choice(['temp', 'torque', 'wear', 'mixed'])
            
            if failure_type == 'temp':
                sample.update({
                    'Air temperature [K]': np.random.normal(305, 4),
                    'Process temperature [K]': np.random.normal(315, 4),
                    'Rotational speed [rpm]': np.random.normal(1500, 120),
                    'Torque [Nm]': np.random.normal(45, 12),
                    'Tool wear [min]': np.random.normal(110, 45)
                })
            elif failure_type == 'torque':
                sample.update({
                    'Air temperature [K]': np.random.normal(300, 2),
                    'Process temperature [K]': np.random.normal(310, 2),
                    'Rotational speed [rpm]': np.random.normal(1400, 150),
                    'Torque [Nm]': np.random.normal(55, 15),
                    'Tool wear [min]': np.random.normal(120, 50)
                })
            elif failure_type == 'wear':
                sample.update({
                    'Air temperature [K]': np.random.normal(301, 2),
                    'Process temperature [K]': np.random.normal(311, 2),
                    'Rotational speed [rpm]': np.random.normal(1500, 100),
                    'Torque [Nm]': np.random.normal(50, 12),
                    'Tool wear [min]': np.random.normal(180, 40)
                })
            else:  # mixed
                sample.update({
                    'Air temperature [K]': np.random.normal(303, 3),
                    'Process temperature [K]': np.random.normal(313, 3),
                    'Rotational speed [rpm]': np.random.normal(1450, 130),
                    'Torque [Nm]': np.random.normal(52, 13),
                    'Tool wear [min]': np.random.normal(140, 50)
                })
    else:
        # Legacy simple generation (kept for backward compatibility)
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
    
    # Ensure realistic bounds
    sample['Air temperature [K]'] = np.clip(sample['Air temperature [K]'], 295, 315)
    sample['Process temperature [K]'] = np.clip(sample['Process temperature [K]'], 305, 330)
    sample['Rotational speed [rpm]'] = np.clip(sample['Rotational speed [rpm]'], 1000, 3000)
    sample['Torque [Nm]'] = np.clip(sample['Torque [Nm]'], 10, 100)
    sample['Tool wear [min]'] = np.clip(sample['Tool wear [min]'], 0, 300)
    
    sample['Fault_Label'] = fault_map[scenario]
    return pd.DataFrame([sample])