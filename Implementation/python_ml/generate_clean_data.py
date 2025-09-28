import numpy as np
import pandas as pd
import os

num_samples = 1000
fault_probability = 0.21  
vibration_length = 100
script_dir = os.path.dirname(__file__)
output_file = os.path.join(script_dir, "clean_synthetic_data.csv")

np.random.seed(42)

data = {"vibration": [], "temp": [], "voltage": [], "label": []}

for _ in range(num_samples):
    is_fault = np.random.random() < fault_probability
    label = 1 if is_fault else 0

    time = np.linspace(0, 10, vibration_length)
    if is_fault:
        vibration = 2.5 * np.sin(3 * time) + 1.5 * np.sin(7 * time) + np.random.normal(0, 0.6, vibration_length)
    else:
        vibration = np.sin(time) + 0.1 * np.sin(5 * time) + np.random.normal(0, 0.15, vibration_length)

    vibration = vibration.tolist()

    temp = np.random.normal(70 if is_fault else 50, 4 if is_fault else 2)
    
    voltage = np.random.normal(245 if is_fault else 220, 8 if is_fault else 4)

    data["vibration"].append(vibration)
    data["temp"].append(temp)
    data["voltage"].append(voltage)
    data["label"].append(label)

df = pd.DataFrame(data)
df.to_csv(output_file, index=False)

# Print statistics
fault_count = df['label'].sum()
normal_count = len(df) - fault_count
print(f"Clean synthetic data generated and saved to {output_file}")
print(f"Total samples: {len(df)}")
print(f"Normal samples: {normal_count} ({normal_count/len(df)*100:.1f}%)")
print(f"Fault samples: {fault_count} ({fault_count/len(df)*100:.1f}%)")
print(f"Temperature range: {df['temp'].min():.1f} - {df['temp'].max():.1f}")
print(f"Voltage range: {df['voltage'].min():.1f} - {df['voltage'].max():.1f}")
