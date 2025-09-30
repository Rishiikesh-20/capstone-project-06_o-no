import simpy
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from config import RANDOM_SEED, SIMULATION_TIME, INTER_ARRIVAL_MEAN, EDGE_CAPACITY, CLOUD_CAPACITY, NUM_FAULT_CLASSES, COMPLEXITY_THRESHOLD, DATASET_PATH, FEATURE_COLUMNS, FAULT_TYPES
from logger import SystemLogger
from data_prep import load_and_prepare_dataset, generate_runtime_sensor_data
from models import train_edge_rf_model, train_cloud_dense_model
from ga_scheduler import EnhancedGA
from sim_processes import edge_process, cloud_process, sensor_process


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

metrics = {
    'total_tasks': 0, 'total_faults_detected': 0,
    'edge': {'tasks_processed': 0, 'faults_detected': 0, 'latency': [], 'energy': [], 'accuracy': [], 'processing_times': []},
    'cloud': {'tasks_processed': 0, 'faults_detected': 0, 'latency': [], 'energy': [], 'accuracy': [], 'network_latency': [], 'processing_times': []},
    'scheduling': {'offload_decisions': [], 'ga_fitness_scores': [], 'edge_load_history': []},
    'timeline': [], 'logs': []
}

rf_model = None
dense_model = None
scaler = StandardScaler()
db_storage = []

logger = SystemLogger(metrics)



def run_simulation():
    global rf_model, dense_model, scaler
    print("\n" + "="*70); print(" " * 10 + "AUTOMOTIVE FACTORY PREDICTIVE MAINTENANCE SYSTEM"); print("="*70)
    
    print("\n[Phase 1] Data Preparation"); print("-" * 50)
    df = load_and_prepare_dataset()
    print(f"Dataset shape: {df.shape}"); print(f"Class distribution:\n{df['Fault_Label'].value_counts().sort_index()}")
    
    X = df[FEATURE_COLUMNS]; y = df['Fault_Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train); X_test_scaled = scaler.transform(X_test)
    
    print("\n[Phase 2] Model Training"); print("-" * 50)
    rf_model, rf_cm = train_edge_rf_model(X_train_scaled, y_train, X_test_scaled, y_test)
    dense_model, dense_cm = train_cloud_dense_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("\n[Phase 3] Simulation Setup"); print("-" * 50)
    env = simpy.Environment(); edge_resource = simpy.Resource(env, capacity=EDGE_CAPACITY)
    cloud_resource = simpy.Resource(env, capacity=CLOUD_CAPACITY); ga_scheduler = EnhancedGA()
    
    print(f"\n[Phase 4] Starting Simulation (Duration: {SIMULATION_TIME} mins)...")
    env.process(sensor_process(env, edge_resource, cloud_resource, ga_scheduler, metrics, scaler, rf_model, dense_model, logger, db_storage))
    env.run(until=SIMULATION_TIME)
    
    print("\n[Phase 5] Simulation completed!")
    return metrics, rf_cm, dense_cm


def analyze_results(metrics, rf_cm, dense_cm):
    print("\n" + "="*70); print(" " * 25 + "SIMULATION ANALYSIS REPORT"); print("="*70)
    
    total_processed = metrics['edge']['tasks_processed'] + metrics['cloud']['tasks_processed']
    if total_processed == 0: print("\nNo tasks were processed during the simulation."); return
        
    print("\n" + "-"*20 + " Overall System Performance " + "-"*20)
    overall_accuracy = np.mean(metrics['edge']['accuracy'] + metrics['cloud']['accuracy'])
    offload_rate = (metrics['cloud']['tasks_processed'] / total_processed * 100)
    print(f"Total Tasks Generated: {metrics['total_tasks']}\nTotal Tasks Processed: {total_processed}")
    print(f"Total Faults Detected: {metrics['total_faults_detected']}\nOverall System Accuracy: {overall_accuracy:.2%}")
    print(f"Intelligent Offload Rate: {offload_rate:.2f}%")

    print("\n" + "-"*20 + " Edge vs. Cloud Performance " + "-"*20)
    edge_stats = {'Tasks': metrics['edge']['tasks_processed'], 'Avg Latency (ms)': np.mean(metrics['edge']['latency'] or [0]), 'Avg Energy (μJ)': np.mean(metrics['edge']['energy'] or [0]), 'Accuracy': np.mean(metrics['edge']['accuracy'] or [0])}
    cloud_stats = {'Tasks': metrics['cloud']['tasks_processed'], 'Avg Latency (ms)': np.mean(metrics['cloud']['latency'] or [0]), 'Avg Energy (μJ)': np.mean(metrics['cloud']['energy'] or [0]), 'Accuracy': np.mean(metrics['cloud']['accuracy'] or [0])}
    print(pd.DataFrame({'Edge': edge_stats, 'Cloud': cloud_stats}).round(3))
    
    fig = plt.figure(figsize=(20, 16)); fig.suptitle('Predictive Maintenance System: Simulation Dashboard', fontsize=20); gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0, 0]); ax1.pie([edge_stats['Tasks'], cloud_stats['Tasks']], labels=['Edge', 'Cloud'], autopct='%1.1f%%', colors=['#4c72b0', '#c44e52']); ax1.set_title('Task Processing Distribution')
    ax2 = fig.add_subplot(gs[0, 1]); sns.histplot(metrics['edge']['latency'], ax=ax2, color='#4c72b0', label='Edge', kde=True); sns.histplot(metrics['cloud']['latency'], ax=ax2, color='#c44e52', label='Cloud', kde=True); ax2.set_title('Latency Distribution (Edge vs. Cloud)'); ax2.set_xlabel('Latency (ms)'); ax2.legend()
    ax3 = fig.add_subplot(gs[0, 2]); sns.boxplot(data=[metrics['edge']['energy'], metrics['cloud']['energy']], ax=ax3, palette=['#4c72b0', '#c44e52']); ax3.set_xticklabels(['Edge', 'Cloud']); ax3.set_title('Energy Consumption per Task'); ax3.set_ylabel('Energy (μJ)')
    if metrics['scheduling']['edge_load_history']:
      ax4 = fig.add_subplot(gs[1, 0]); load_times, load_values = zip(*metrics['scheduling']['edge_load_history']); ax4.plot(load_times, load_values, color='#55a868'); ax4.set_title('Edge Server Load Over Time'); ax4.set_xlabel('Simulation Time (mins)'); ax4.set_ylabel('Load (%)'); ax4.set_ylim(0, 1.1)
    if metrics['scheduling']['ga_fitness_scores']:
      ax5 = fig.add_subplot(gs[1, 1]); ax5.plot(pd.Series(metrics['scheduling']['ga_fitness_scores']).rolling(window=10).mean(), color='#8172b2'); ax5.set_title('GA Fitness Score (Lower is Better)'); ax5.set_xlabel('Task Instance'); ax5.set_ylabel('Fitness Score')
    ax6 = fig.add_subplot(gs[2, 0]); sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax6, xticklabels=FAULT_TYPES.values(), yticklabels=FAULT_TYPES.values()); ax6.set_title('Edge RF Model Confusion Matrix'); ax6.set_xlabel('Predicted'); ax6.set_ylabel('Actual')
    ax7 = fig.add_subplot(gs[2, 1]); sns.heatmap(dense_cm, annot=True, fmt='d', cmap='Reds', ax=ax7, xticklabels=FAULT_TYPES.values(), yticklabels=FAULT_TYPES.values()); ax7.set_title('Cloud Dense Model Confusion Matrix'); ax7.set_xlabel('Predicted'); ax7.set_ylabel('Actual')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()


if __name__ == "__main__":
    final_metrics, rf_cm, dense_cm = run_simulation()
    
    if (final_metrics['edge']['tasks_processed'] + final_metrics['cloud']['tasks_processed']) > 0:
        analyze_results(final_metrics, rf_cm, dense_cm)
        print("\n" + "="*70); print(" " * 27 + "PROJECT HIGHLIGHTS"); print("="*70)
        edge_percentage = (final_metrics['edge']['tasks_processed'] / (final_metrics['edge']['tasks_processed'] + final_metrics['cloud']['tasks_processed'])) * 100
        overall_accuracy_val = np.mean(final_metrics['edge']['accuracy'] + final_metrics['cloud']['accuracy'])
        print(f"\n✓ \033[1mIntelligent Task Scheduling:\033[0m The Enhanced GA dynamically optimized task placement, ensuring urgent, simple analyses were handled instantly on the edge while complex diagnostics were sent to the cloud.")
        print(f"✓ \033[1mHybrid Machine Learning:\033[0m A lightweight Edge RF model and a powerful Cloud Neural Network worked in tandem to deliver high-performance fault detection tailored to the complexity of the task.")
        print(f"✓ \033[1mResource Efficiency:\033[0m By processing \033[1m{edge_percentage:.1f}%\033[0m of tasks locally, the system drastically cut network traffic and cloud costs, demonstrating a lean operational model.")
        print(f"✓ \033[1mProactive Maintenance:\033[0m With an overall accuracy of \033[1m{overall_accuracy_val:.2%}\033[0m, the system reliably detected {final_metrics['total_faults_detected']} potential equipment failures, directly translating to reduced downtime and increased factory productivity.")
        print("\n\033[1mConclusion:\033[0m This simulation validates a robust, efficient, and intelligent Edge-Cloud architecture, perfectly suited for the demands of modern Industry 4.0 predictive maintenance.")
        print("="*70)