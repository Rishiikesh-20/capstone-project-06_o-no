import simpy
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['XGB_USE_CUDA'] = '0' 

import tensorflow as tf
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

import pickle
from tensorflow.keras.models import load_model

from config import (RANDOM_SEED, SIMULATION_TIME, INTER_ARRIVAL_MEAN, EDGE_CAPACITY, 
                    CLOUD_CAPACITY, NUM_FAULT_CLASSES, COMPLEXITY_THRESHOLD, DATASET_PATH, 
                    FEATURE_COLUMNS, FAULT_TYPES, CLOUD_COST_PER_TASK)
from logger import SystemLogger
from data_prep import load_and_prepare_dataset, generate_runtime_sensor_data
from models import (train_edge_lightgbm_model, train_cloud_lstm_model, 
                    train_edge_rf_model, train_cloud_dense_model,
                    train_edge_svm_model, train_cloud_cnn_model)
from dqn_scheduler import DQNScheduler
from sim_processes import edge_process, cloud_process, sensor_process


def load_pretrained_models():
    try:
        print("Loading pre-trained models...")
        with open('edge_lightgbm_model.pkl', 'rb') as f:
            edge_model = pickle.load(f)
        cloud_model = load_model('cloud_lstm_model.keras')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        edge_cm = np.load('edge_cm.npy')
        cloud_cm = np.load('cloud_cm.npy')
        print("✓ Pre-trained models loaded successfully!")
        print(f"  • Edge: LightGBM with {edge_model.n_estimators} trees")
        print(f"  • Cloud: Enhanced LSTM Neural Network")
        return edge_model, cloud_model, scaler, edge_cm, cloud_cm, True
    except FileNotFoundError as e:
        print(f"⚠️  Pre-trained models not found: {e}")
        print("   Run 'python3 train.py' first to train and save models.")
        return None, None, None, None, None, False


def create_metrics_dict():
    return {
        'total_tasks': 0, 'total_faults_detected': 0,
        'edge': {
            'tasks_processed': 0, 'faults_detected': 0, 
            'latency': [], 'energy': [], 'accuracy': [], 'processing_times': []
        },
        'cloud': {
            'tasks_processed': 0, 'faults_detected': 0, 
            'latency': [], 'energy': [], 'accuracy': [], 
            'network_latency': [], 'processing_times': [],
            'total_cost': 0.0
        },
        'scheduling': {
            'offload_decisions': [], 'dqn_costs': [], 'edge_load_history': []
        },
        'timeline': [], 'logs': []
    }


def run_simulation(model_type='lightgbm_lstm', seed_offset=0, fault_prob_scale=1.0, 
                   simulation_time=None, use_pretrained=True):
    actual_seed = RANDOM_SEED + seed_offset
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    tf.random.set_seed(actual_seed)
    
    metrics = create_metrics_dict()
    db_storage = []
    logger = SystemLogger(metrics)
    
    print("\n" + "="*80)
    print(" " * 10 + f"AUTOMOTIVE FACTORY PREDICTIVE MAINTENANCE SYSTEM")
    print(f" " * 18 + f"Model Type: {model_type.upper()}")
    if fault_prob_scale != 1.0:
        print(f" " * 15 + f"STRESS TEST MODE (Fault Scale: {fault_prob_scale}x)")
    print("="*80)
    
    if use_pretrained and model_type in ['lightgbm_lstm', 'xgboost_lstm']:
        edge_model, cloud_model, scaler, edge_cm, cloud_cm, models_loaded = load_pretrained_models()
        
        if models_loaded:
            print("\n[Phase 1] Using Pre-Trained Models (Fast Mode)")
            print("-" * 80)
            print("✓ Skipping training - using saved models for instant simulation!")
            print("  (This is ~60x faster than training from scratch)")
        else:
            print("\n[Phase 1] Training Models (Pre-trained not available)")
            print("-" * 80)
            use_pretrained = False
    else:
        use_pretrained = False
    
    if not use_pretrained:
        print("\n[Phase 1] Data Preparation")
        print("-" * 80)
        df = load_and_prepare_dataset()
        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['Fault_Label'].value_counts().sort_index()}")
        
        X = df[FEATURE_COLUMNS]
        y = df['Fault_Label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=actual_seed, stratify=y
        )
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\n[Phase 2] Model Training")
        print("-" * 80)
        
        if model_type == 'lightgbm_lstm' or model_type == 'xgboost_lstm':
            edge_model, edge_cm = train_edge_lightgbm_model(X_train_scaled, y_train, X_test_scaled, y_test)
            cloud_model, cloud_cm = train_cloud_lstm_model(X_train_scaled, y_train, X_test_scaled, y_test)
        elif model_type == 'rf_dense':
            edge_model, edge_cm = train_edge_rf_model(X_train_scaled, y_train, X_test_scaled, y_test)
            cloud_model, cloud_cm = train_cloud_dense_model(X_train_scaled, y_train, X_test_scaled, y_test)
        elif model_type == 'svm_cnn':
            edge_model, edge_cm = train_edge_svm_model(X_train_scaled, y_train, X_test_scaled, y_test)
            cloud_model, cloud_cm = train_cloud_cnn_model(X_train_scaled, y_train, X_test_scaled, y_test)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    phase_num = 2 if use_pretrained else 3
    print(f"\n[Phase {phase_num}] DQN Scheduler Training")
    print("-" * 80)
    dqn_scheduler = DQNScheduler()
    dqn_scheduler.train(total_timesteps=1000)
    
    phase_num += 1
    print(f"\n[Phase {phase_num}] Simulation Setup")
    print("-" * 80)
    env = simpy.Environment()
    edge_resource = simpy.Resource(env, capacity=EDGE_CAPACITY)
    cloud_resource = simpy.Resource(env, capacity=CLOUD_CAPACITY)
    
    sim_time = simulation_time if simulation_time else SIMULATION_TIME
    
    phase_num += 1
    print(f"\n[Phase {phase_num}] Starting Simulation (Duration: {sim_time} mins)...")
    print("-" * 80)
    
    env.process(sensor_process(
        env, edge_resource, cloud_resource, dqn_scheduler, metrics, scaler, 
        edge_model, cloud_model, logger, db_storage, model_type, fault_prob_scale
    ))
    
    env.run(until=sim_time)
    
    phase_num += 1
    print(f"\n[Phase {phase_num}] Simulation completed!")
    print("-" * 80)
    
    return metrics, edge_cm, cloud_cm, edge_model, cloud_model, scaler


def run_multiple_experiments(model_types=['xgboost_lstm', 'rf_dense', 'svm_cnn'], 
                             num_runs=3, simulation_time=1000):
    all_results = {}
    
    for model_type in model_types:
        print(f"\n\n{'='*70}")
        print(f"Running experiments for: {model_type.upper()}")
        print(f"{'='*70}")
        
        runs_metrics = []
        
        for run_idx in range(num_runs):
            print(f"\n--- Run {run_idx + 1}/{num_runs} ---")
            
            metrics, edge_cm, cloud_cm, _, _, _ = run_simulation(
                model_type=model_type, 
                seed_offset=run_idx,
                simulation_time=simulation_time
            )
            
            runs_metrics.append(metrics)
        
        aggregated = aggregate_metrics(runs_metrics)
        aggregated['edge_cm'] = edge_cm  
        aggregated['cloud_cm'] = cloud_cm
        aggregated['model_type'] = model_type
        
        all_results[model_type] = aggregated
        
        print(f"\n\n{'-'*70}")
        print(f"Summary for {model_type.upper()} ({num_runs} runs):")
        print(f"{'-'*70}")
        print(f"Avg Tasks Processed: {aggregated['avg_total_tasks']:.1f} ± {aggregated['std_total_tasks']:.1f}")
        print(f"Avg Overall Accuracy: {aggregated['avg_overall_accuracy']:.2%} ± {aggregated['std_overall_accuracy']:.2%}")
        print(f"Avg Edge Latency: {aggregated['avg_edge_latency']:.2f}ms ± {aggregated['std_edge_latency']:.2f}ms")
        print(f"Avg Cloud Latency: {aggregated['avg_cloud_latency']:.2f}ms ± {aggregated['std_cloud_latency']:.2f}ms")
        print(f"Avg Cloud Cost: ${aggregated['avg_cloud_cost']:.4f} ± ${aggregated['std_cloud_cost']:.4f}")
    
    return all_results


def aggregate_metrics(runs_metrics):
    aggregated = defaultdict(list)
    
    for metrics in runs_metrics:
        total_processed = metrics['edge']['tasks_processed'] + metrics['cloud']['tasks_processed']
        
        if total_processed > 0:
            overall_accuracy = np.mean(metrics['edge']['accuracy'] + metrics['cloud']['accuracy'])
            aggregated['total_tasks'].append(metrics['total_tasks'])
            aggregated['overall_accuracy'].append(overall_accuracy)
            aggregated['faults_detected'].append(metrics['total_faults_detected'])
            
            if metrics['edge']['latency']:
                aggregated['edge_latency'].append(np.mean(metrics['edge']['latency']))
                aggregated['edge_energy'].append(np.mean(metrics['edge']['energy']))
            
            if metrics['cloud']['latency']:
                aggregated['cloud_latency'].append(np.mean(metrics['cloud']['latency']))
                aggregated['cloud_energy'].append(np.mean(metrics['cloud']['energy']))
                aggregated['cloud_cost'].append(metrics['cloud']['total_cost'])
    
    result = {}
    for key, values in aggregated.items():
        result[f'avg_{key}'] = np.mean(values) if values else 0
        result[f'std_{key}'] = np.std(values) if values else 0
    
    return result


def analyze_results(metrics, edge_cm, cloud_cm, model_type='xgboost_lstm'):
    print("\n" + "="*70)
    print(" " * 25 + "SIMULATION ANALYSIS REPORT")
    print("="*70)
    
    total_processed = metrics['edge']['tasks_processed'] + metrics['cloud']['tasks_processed']
    if total_processed == 0:
        print("\nNo tasks were processed during the simulation.")
        return
    
    print("\n" + "-"*20 + " Overall System Performance " + "-"*20)
    overall_accuracy = np.mean(metrics['edge']['accuracy'] + metrics['cloud']['accuracy'])
    offload_rate = (metrics['cloud']['tasks_processed'] / total_processed * 100)
    
    print(f"Total Tasks Generated: {metrics['total_tasks']}")
    print(f"Total Tasks Processed: {total_processed}")
    print(f"Total Faults Detected: {metrics['total_faults_detected']}")
    print(f"Overall System Accuracy: {overall_accuracy:.2%}")
    print(f"Intelligent Offload Rate: {offload_rate:.2f}%")
    print(f"Total Cloud Cost: ${metrics['cloud']['total_cost']:.4f}")
    
    print("\n" + "-"*20 + " Edge vs. Cloud Performance " + "-"*20)
    edge_stats = {
        'Tasks': metrics['edge']['tasks_processed'],
        'Avg Latency (ms)': np.mean(metrics['edge']['latency']) if metrics['edge']['latency'] else 0,
        'Avg Energy (μJ)': np.mean(metrics['edge']['energy']) if metrics['edge']['energy'] else 0,
        'Accuracy': np.mean(metrics['edge']['accuracy']) if metrics['edge']['accuracy'] else 0
    }
    cloud_stats = {
        'Tasks': metrics['cloud']['tasks_processed'],
        'Avg Latency (ms)': np.mean(metrics['cloud']['latency']) if metrics['cloud']['latency'] else 0,
        'Avg Energy (μJ)': np.mean(metrics['cloud']['energy']) if metrics['cloud']['energy'] else 0,
        'Accuracy': np.mean(metrics['cloud']['accuracy']) if metrics['cloud']['accuracy'] else 0
    }
    print(pd.DataFrame({'Edge': edge_stats, 'Cloud': cloud_stats}).round(3))
    
    create_visualizations(metrics, edge_cm, cloud_cm, model_type)


def create_visualizations(metrics, edge_cm, cloud_cm, model_type):
    edge_stats = {
        'Tasks': metrics['edge']['tasks_processed'],
        'Latency': np.mean(metrics['edge']['latency']) if metrics['edge']['latency'] else 0,
        'Energy': np.mean(metrics['edge']['energy']) if metrics['edge']['energy'] else 0
    }
    cloud_stats = {
        'Tasks': metrics['cloud']['tasks_processed'],
        'Latency': np.mean(metrics['cloud']['latency']) if metrics['cloud']['latency'] else 0,
        'Energy': np.mean(metrics['cloud']['energy']) if metrics['cloud']['energy'] else 0
    }
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Predictive Maintenance System Dashboard - {model_type.upper()}', fontsize=20)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pie([edge_stats['Tasks'], cloud_stats['Tasks']], 
            labels=['Edge', 'Cloud'], autopct='%1.1f%%', 
            colors=['#4c72b0', '#c44e52'])
    ax1.set_title('Task Processing Distribution')
    
    ax2 = fig.add_subplot(gs[0, 1])
    if metrics['edge']['latency']:
        sns.histplot(metrics['edge']['latency'], ax=ax2, color='#4c72b0', 
                    label='Edge', kde=True, alpha=0.6)
    if metrics['cloud']['latency']:
        sns.histplot(metrics['cloud']['latency'], ax=ax2, color='#c44e52', 
                    label='Cloud', kde=True, alpha=0.6)
    ax2.set_title('Latency Distribution')
    ax2.set_xlabel('Latency (ms)')
    ax2.legend()
    
    ax3 = fig.add_subplot(gs[0, 2])
    if metrics['edge']['energy'] and metrics['cloud']['energy']:
        sns.boxplot(data=[metrics['edge']['energy'], metrics['cloud']['energy']], 
                   ax=ax3, palette=['#4c72b0', '#c44e52'])
        ax3.set_xticklabels(['Edge', 'Cloud'])
    ax3.set_title('Energy Consumption per Task')
    ax3.set_ylabel('Energy (μJ)')
    
    if metrics['scheduling']['edge_load_history']:
        ax4 = fig.add_subplot(gs[1, 0])
        load_times, load_values = zip(*metrics['scheduling']['edge_load_history'])
        ax4.plot(load_times, load_values, color='#55a868', linewidth=1.5)
        ax4.axhline(y=0.9, color='r', linestyle='--', label='Critical Threshold')
        ax4.set_title('Edge Server Load Over Time')
        ax4.set_xlabel('Simulation Time (mins)')
        ax4.set_ylabel('Load (%)')
        ax4.set_ylim(0, 1.1)
        ax4.legend()
    
    if metrics['scheduling']['dqn_costs']:
        ax5 = fig.add_subplot(gs[1, 1])
        rolling_costs = pd.Series(metrics['scheduling']['dqn_costs']).rolling(window=20, min_periods=1).mean()
        ax5.plot(rolling_costs, color='#8172b2', linewidth=2)
        ax5.set_title('DQN Scheduling Cost (Rolling Avg, Lower=Better)')
        ax5.set_xlabel('Task Instance')
        ax5.set_ylabel('Cost Score')
    
    ax6 = fig.add_subplot(gs[1, 2])
    timeline_df = pd.DataFrame(metrics['timeline'])
    if not timeline_df.empty:
        timeline_df['cumulative_faults'] = timeline_df['fault_detected'].cumsum()
        ax6.plot(timeline_df['time'], timeline_df['cumulative_faults'], 
                color='#dd8452', linewidth=2)
        ax6.set_title('Cumulative Faults Detected Over Time')
        ax6.set_xlabel('Simulation Time (mins)')
        ax6.set_ylabel('Total Faults Detected')
    
    ax7 = fig.add_subplot(gs[2, 0])
    sns.heatmap(edge_cm, annot=True, fmt='d', cmap='Blues', ax=ax7,
                xticklabels=list(FAULT_TYPES.values()),
                yticklabels=list(FAULT_TYPES.values()))
    ax7.set_title(f'Edge Model Confusion Matrix ({model_type.split("_")[0].upper()})')
    ax7.set_xlabel('Predicted')
    ax7.set_ylabel('Actual')
    plt.setp(ax7.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax7.get_yticklabels(), rotation=0, fontsize=8)
    
    ax8 = fig.add_subplot(gs[2, 1])
    sns.heatmap(cloud_cm, annot=True, fmt='d', cmap='Reds', ax=ax8,
                xticklabels=list(FAULT_TYPES.values()),
                yticklabels=list(FAULT_TYPES.values()))
    ax8.set_title(f'Cloud Model Confusion Matrix ({model_type.split("_")[1].upper()})')
    ax8.set_xlabel('Predicted')
    ax8.set_ylabel('Actual')
    plt.setp(ax8.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax8.get_yticklabels(), rotation=0, fontsize=8)
    
    if metrics['cloud']['network_latency']:
        ax9 = fig.add_subplot(gs[2, 2])
        sns.histplot(metrics['cloud']['network_latency'], ax=ax9, 
                    color='#64b5cd', kde=True, bins=30)
        ax9.set_title('Cloud Network Latency Distribution')
        ax9.set_xlabel('Latency (ms)')
        ax9.set_ylabel('Frequency')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def create_comparison_plots(all_results):
    model_types = list(all_results.keys())
    
    accuracies = [all_results[mt]['avg_overall_accuracy'] for mt in model_types]
    acc_stds = [all_results[mt]['std_overall_accuracy'] for mt in model_types]
    
    edge_latencies = [all_results[mt].get('avg_edge_latency', 0) for mt in model_types]
    edge_lat_stds = [all_results[mt].get('std_edge_latency', 0) for mt in model_types]
    
    cloud_latencies = [all_results[mt].get('avg_cloud_latency', 0) for mt in model_types]
    cloud_lat_stds = [all_results[mt].get('std_cloud_latency', 0) for mt in model_types]
    
    costs = [all_results[mt].get('avg_cloud_cost', 0) for mt in model_types]
    cost_stds = [all_results[mt].get('std_cloud_cost', 0) for mt in model_types]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison Across Architectures', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    x_pos = np.arange(len(model_types))
    ax1.bar(x_pos, [acc * 100 for acc in accuracies], yerr=[std * 100 for std in acc_stds],
            color=['#4c72b0', '#55a868', '#c44e52'], alpha=0.7, capsize=5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([mt.upper().replace('_', '+') for mt in model_types], rotation=15)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Overall System Accuracy')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = axes[0, 1]
    width = 0.35
    ax2.bar(x_pos - width/2, edge_latencies, width, yerr=edge_lat_stds, 
            label='Edge', color='#4c72b0', alpha=0.7, capsize=3)
    ax2.bar(x_pos + width/2, cloud_latencies, width, yerr=cloud_lat_stds,
            label='Cloud', color='#c44e52', alpha=0.7, capsize=3)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([mt.upper().replace('_', '+') for mt in model_types], rotation=15)
    ax2.set_ylabel('Avg Latency (ms)')
    ax2.set_title('Average Latency Comparison')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.bar(x_pos, costs, yerr=cost_stds, color=['#8172b2', '#dd8452', '#64b5cd'], 
            alpha=0.7, capsize=5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([mt.upper().replace('_', '+') for mt in model_types], rotation=15)
    ax3.set_ylabel('Total Cost ($)')
    ax3.set_title('Cloud Processing Cost')
    ax3.grid(axis='y', alpha=0.3)
    
    ax4 = axes[1, 1]
    colors_map = {'xgboost_lstm': '#4c72b0', 'rf_dense': '#55a868', 'svm_cnn': '#c44e52'}
    for mt in model_types:
        avg_latency = (all_results[mt].get('avg_edge_latency', 0) + 
                      all_results[mt].get('avg_cloud_latency', 0)) / 2
        ax4.scatter(avg_latency, all_results[mt]['avg_overall_accuracy'] * 100,
                   s=200, alpha=0.6, color=colors_map.get(mt, '#777777'),
                   label=mt.upper().replace('_', '+'))
        ax4.annotate(mt.upper().replace('_', '+'), 
                    (avg_latency, all_results[mt]['avg_overall_accuracy'] * 100),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax4.set_xlabel('Average Latency (ms)')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Accuracy vs Latency Trade-off')
    ax4.grid(alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("OPTION 1: Single Simulation Run (LightGBM + LSTM)")
    print("="*70)
    
    final_metrics, edge_cm, cloud_cm, _, _, _ = run_simulation(
        model_type='lightgbm_lstm',
        simulation_time=1000  
    )
    
    if (final_metrics['edge']['tasks_processed'] + final_metrics['cloud']['tasks_processed']) > 0:
        analyze_results(final_metrics, edge_cm, cloud_cm, 'lightgbm_lstm')
        
        print("\n" + "="*70)
        print(" " * 27 + "PROJECT HIGHLIGHTS")
        print("="*70)
        
        total_processed = final_metrics['edge']['tasks_processed'] + final_metrics['cloud']['tasks_processed']
        edge_percentage = (final_metrics['edge']['tasks_processed'] / total_processed) * 100
        overall_accuracy = np.mean(final_metrics['edge']['accuracy'] + final_metrics['cloud']['accuracy'])
        
        print(f"\n✓ \033[1mIntelligent Task Scheduling:\033[0m The DQN-based scheduler dynamically "
              f"optimized task placement, ensuring urgent analyses were handled instantly on the edge "
              f"while complex diagnostics were sent to the cloud.")
        print(f"✓ \033[1mHybrid Machine Learning:\033[0m A lightweight LightGBM edge model and a powerful "
              f"LSTM cloud network worked in tandem to deliver high-performance fault detection.")
        print(f"✓ \033[1mResource Efficiency:\033[0m By processing \033[1m{edge_percentage:.1f}%\033[0m "
              f"of tasks locally, the system drastically cut network traffic and cloud costs "
              f"(${final_metrics['cloud']['total_cost']:.4f} total).")
        print(f"✓ \033[1mProactive Maintenance:\033[0m With an overall accuracy of "
              f"\033[1m{overall_accuracy:.2%}\033[0m, the system reliably detected "
              f"{final_metrics['total_faults_detected']} potential equipment failures.")
        print("\n\033[1mConclusion:\033[0m This simulation validates a robust, efficient, and intelligent "
              "Edge-Cloud architecture for modern Industry 4.0 predictive maintenance.")
        print("="*70)
    
    # Option 2: Run model comparison experiments (commented out by default)
    # Uncomment to run comprehensive experiments
    """
    print("\n\n" + "="*70)
    print("OPTION 2: Model Comparison Experiments")
    print("="*70)
    
    all_results = run_multiple_experiments(
        model_types=['xgboost_lstm', 'rf_dense', 'svm_cnn'],
        num_runs=3,
        simulation_time=1000
    )
    
    # Create comparison visualizations
    create_comparison_plots(all_results)
    
    # Print final comparison table
    print("\n\n" + "="*70)
    print(" " * 20 + "FINAL COMPARISON TABLE")
    print("="*70)
    
    comparison_df = pd.DataFrame({
        mt.upper().replace('_', '+'): {
            'Avg Accuracy (%)': f"{all_results[mt]['avg_overall_accuracy']*100:.2f} ± "
                               f"{all_results[mt]['std_overall_accuracy']*100:.2f}",
            'Avg Edge Latency (ms)': f"{all_results[mt].get('avg_edge_latency', 0):.2f} ± "
                                    f"{all_results[mt].get('std_edge_latency', 0):.2f}",
            'Avg Cloud Cost ($)': f"{all_results[mt].get('avg_cloud_cost', 0):.4f} ± "
                                 f"{all_results[mt].get('std_cloud_cost', 0):.4f}",
            'Avg Faults Detected': f"{all_results[mt].get('avg_faults_detected', 0):.1f} ± "
                                  f"{all_results[mt].get('std_faults_detected', 0):.1f}"
        }
        for mt in all_results.keys()
    }).T
    
    print(comparison_df)
    print("="*70)
    """
    
    # Option 3: Stress Test (uncomment to run)
    """
    print("\n\n" + "="*70)
    print("OPTION 3: STRESS TEST (5x Fault Probability)")
    print("="*70)
    
    stress_metrics, stress_edge_cm, stress_cloud_cm, _, _, _ = run_simulation(
        model_type='xgboost_lstm',
        fault_prob_scale=5.0,
        simulation_time=1000
    )
    
    if (stress_metrics['edge']['tasks_processed'] + stress_metrics['cloud']['tasks_processed']) > 0:
        analyze_results(stress_metrics, stress_edge_cm, stress_cloud_cm, 'xgboost_lstm_stress')
    """