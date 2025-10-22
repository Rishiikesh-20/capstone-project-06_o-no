"""
Fixed MQTT Integration for Predictive Maintenance System
Solves timeout issues with intelligent buffering
"""

import simpy
import numpy as np
import pandas as pd
import time
import random
from mqtt_sensor_handler import MQTTSensorHandler
from config import (FEATURE_COLUMNS, FAULT_TYPES, COMPLEXITY_THRESHOLD, 
                    NET_LATENCY_MEAN, NET_LATENCY_STD, CLOUD_COST_PER_TASK,
                    EDGE_CAPACITY, INTER_ARRIVAL_MEAN)


class MQTTSensorBridge:
    """
    Improved MQTT-SimPy Bridge with intelligent buffering
    Handles slow sensor publishing rates gracefully
    """
    
    def __init__(self, mqtt_handler, use_mqtt=True):
        self.mqtt_handler = mqtt_handler
        self.use_mqtt = use_mqtt
        self.sensor_buffer = {}
        self.last_mqtt_time = time.time()
        self.mqtt_data_cache = []  
        self.cache_size = 50
        self.stats = {
            'mqtt_readings': 0,
            'synthetic_readings': 0,
            'cache_hits': 0,
            'timeout_count': 0
        }
        
        self.sensor_mapping = {
            'temperature': 'Air temperature [K]',
            'temp': 'Air temperature [K]',
            'process_temp': 'Process temperature [K]',
            'rpm': 'Rotational speed [rpm]',
            'speed': 'Rotational speed [rpm]',
            'torque': 'Torque [Nm]',
            'tool_wear': 'Tool wear [min]',
            'wear': 'Tool wear [min]',
            'vibration': 'Vibration',  
            'pressure': 'Pressure'      
        }
        
        print(f"✓ MQTT Bridge initialized (Mode: {'MQTT' if use_mqtt else 'Synthetic'})")
        print(f"  • Cache size: {self.cache_size} readings")
        print(f"  • Intelligent fallback enabled")
    
    def get_sensor_reading(self, timeout=0.1):
        """
        Get sensor reading with intelligent fallback
        Now much faster and handles slow publishers
        """
        if not self.use_mqtt or not self.mqtt_handler.connected:
            self.stats['synthetic_readings'] += 1
            return self._generate_synthetic_data()
        
        mqtt_data = self.mqtt_handler.get_sensor_data(timeout=timeout)
        
        if mqtt_data:
            self.last_mqtt_time = time.time()
            df = self._convert_mqtt_to_dataframe(mqtt_data)
            
            if len(self.mqtt_data_cache) < self.cache_size:
                self.mqtt_data_cache.append(df.copy())
            
            self.stats['mqtt_readings'] += 1
            return df
        
        if self.mqtt_data_cache and (time.time() - self.last_mqtt_time < 10.0):
            cached = random.choice(self.mqtt_data_cache)
            varied = self._add_variation_to_cached(cached)
            self.stats['cache_hits'] += 1
            return varied
        
        if time.time() - self.last_mqtt_time > 10.0:
            if self.stats['timeout_count'] % 10 == 0: 
                print(f"⚠ MQTT data stale (>10s) - using synthetic (count: {self.stats['timeout_count']})")
            self.stats['timeout_count'] += 1
        
        self.stats['synthetic_readings'] += 1
        return self._generate_synthetic_data()
    
    def _convert_mqtt_to_dataframe(self, mqtt_data):
        """Convert single MQTT message to DataFrame"""
        sensor_type = mqtt_data['data'].get('sensor_type', '').lower()
        value = mqtt_data['data'].get('value', 0.0)
        
        feature_name = self.sensor_mapping.get(sensor_type)
        
        sample = self._get_default_sample()
        
        if feature_name and feature_name in sample:
            sample[feature_name] = value
        
        sample['Fault_Label'] = self._detect_fault_from_values(sample)
        
        return pd.DataFrame([sample])
    
    def _get_default_sample(self):
        """Get default sensor values"""
        return {
            'Air temperature [K]': 300.0,
            'Process temperature [K]': 310.0,
            'Rotational speed [rpm]': 1500.0,
            'Torque [Nm]': 40.0,
            'Tool wear [min]': 100.0
        }
    
    def _add_variation_to_cached(self, cached_df):
        """Add small random variation to cached data"""
        varied = cached_df.copy()
        
        for col in FEATURE_COLUMNS:
            if col in varied.columns:
                original_value = varied[col].iloc[0]
                
                variation = original_value * np.random.uniform(-0.03, 0.03)
                varied.loc[0, col] = original_value + variation
        
        sample = varied.iloc[0].to_dict()
        varied.loc[0, 'Fault_Label'] = self._detect_fault_from_values(sample)
        
        return varied
    
    def _detect_fault_from_values(self, sample):
        """Heuristic fault detection from sensor values"""
        
        if sample.get('Tool wear [min]', 0) > 200:
            return 1
        
        if sample.get('Air temperature [K]', 0) > 305 or \
           sample.get('Process temperature [K]', 0) > 320:
            return 2
        
        if sample.get('Rotational speed [rpm]', 0) < 1200:
            return 3
        
        if sample.get('Torque [Nm]', 0) > 60:
            return 4
        
        if np.random.random() < 0.05:
            return 5
        
        return 0 
    
    def _generate_synthetic_data(self):
        """Generate synthetic data (fallback)"""
        from data_prep import generate_runtime_sensor_data
        scenarios = list(FAULT_TYPES.values())
        scenario_probs = [0.7, 0.05, 0.05, 0.05, 0.05, 0.1]
        scenario = np.random.choice(scenarios, p=scenario_probs)
        return generate_runtime_sensor_data(scenario)
    
    def get_statistics(self):
        """Get bridge statistics"""
        total = self.stats['mqtt_readings'] + self.stats['synthetic_readings'] + self.stats['cache_hits']
        return {
            'mqtt_readings': self.stats['mqtt_readings'],
            'synthetic_readings': self.stats['synthetic_readings'],
            'cache_hits': self.stats['cache_hits'],
            'timeout_count': self.stats['timeout_count'],
            'mqtt_percentage': (self.stats['mqtt_readings'] / total * 100) if total > 0 else 0,
            'cache_percentage': (self.stats['cache_hits'] / total * 100) if total > 0 else 0
        }


def mqtt_sensor_process(env, edge_res, cloud_res, dqn_scheduler, metrics, scaler,
                        edge_model, cloud_model, logger, db_storage, model_type,
                        mqtt_bridge, fault_prob_scale=1.0):
    """
    Modified sensor process using MQTT data with intelligent buffering
    """
    task_id = 0
    
    while True:
        data = mqtt_bridge.get_sensor_reading(timeout=0.1)
        
        if data is None or data.empty:
            yield env.timeout(0.5)
            continue
        
        task = {
            'id': task_id,
            'data': data,
            'arrival': env.now,
            'complexity': random.uniform(0.5, 3),
            'true_label': data['Fault_Label'].iloc[0]
        }
        
        task_id += 1
        metrics['total_tasks'] += 1
        
        env.process(edge_process(
            env, task, edge_res, cloud_res, dqn_scheduler,
            metrics, scaler, edge_model, cloud_model, logger,
            db_storage, model_type
        ))
        
        yield env.timeout(random.expovariate(1/INTER_ARRIVAL_MEAN))


def edge_process(env, task, edge_res, cloud_res, dqn_scheduler, metrics, scaler,
                 edge_model, cloud_model, logger, db_storage, model_type='lightgbm_lstm'):
    """Your existing edge_process - no changes"""
    with edge_res.request() as req:
        yield req
        edge_load = min((len(edge_res.users) + len(edge_res.queue)) / EDGE_CAPACITY, 1.0)
        metrics['scheduling']['edge_load_history'].append((env.now, edge_load))
        
        logger.check_and_alert_high_load(env.now, edge_load)
        
        features = task['data'][FEATURE_COLUMNS].values[0]
        features_normalized = scaler.transform([features])
        
        preprocess_time = 1.5 + random.uniform(0, 0.5)
        yield env.timeout(preprocess_time)
        
        net_latency = max(0.1, np.random.normal(NET_LATENCY_MEAN, NET_LATENCY_STD))
        decision, dqn_cost = dqn_scheduler.schedule(
            task['complexity'], edge_load, len(cloud_res.queue), net_latency
        )
        metrics['scheduling']['dqn_costs'].append(dqn_cost)
        metrics['scheduling']['offload_decisions'].append({
            'decision': 'edge' if decision == 0 else 'cloud',
            'complexity': task['complexity'],
            'edge_load': edge_load
        })
        
        if decision == 0 and task['complexity'] < COMPLEXITY_THRESHOLD:
            pred = edge_model.predict(features_normalized)[0]
            confidence = np.max(edge_model.predict_proba(features_normalized)[0])
            
            inference_time = 0.5 + task['complexity'] * 0.3
            yield env.timeout(inference_time)
            
            total_latency = env.now - task['arrival']
            energy_consumed = task['complexity'] * 0.5 + preprocess_time * 0.1
            accuracy = 1.0 if pred == task['true_label'] else 0.0
            
            metrics['edge']['tasks_processed'] += 1
            metrics['edge']['latency'].append(total_latency)
            metrics['edge']['energy'].append(energy_consumed)
            metrics['edge']['accuracy'].append(accuracy)
            metrics['edge']['processing_times'].append(inference_time)
            
            if pred != 0:
                metrics['edge']['faults_detected'] += 1
                metrics['total_faults_detected'] += 1
                logger.log_event(env.now, 'FAULT_DETECTED',
                               f"EDGE detected {FAULT_TYPES[pred]} (confidence: {confidence:.2f})")
            
            logger.log_task_completion(env.now, 'edge', task['id'], total_latency,
                                     energy_consumed, accuracy, pred != 0)
        else:
            logger.log_event(env.now, 'OFFLOAD_DECISION',
                           f"Task {task['id']} offloaded to cloud (Complexity: {task['complexity']:.2f}, "
                           f"Edge Load: {edge_load:.2f})")
            task['preprocessed_features'] = features_normalized
            task['net_latency'] = net_latency
            env.process(cloud_process(env, task, cloud_res, metrics, scaler,
                                    cloud_model, logger, db_storage, model_type))


def cloud_process(env, task, cloud_res, metrics, scaler, cloud_model, logger,
                  db_storage, model_type='lightgbm_lstm'):
    """Your existing cloud_process - no changes"""
    with cloud_res.request() as req:
        yield req
        
        network_latency = task.get('net_latency',
                                   max(0.1, np.random.normal(NET_LATENCY_MEAN, NET_LATENCY_STD)))
        yield env.timeout(network_latency)
        metrics['cloud']['network_latency'].append(network_latency)
        
        features_normalized = task.get('preprocessed_features')
        if features_normalized is None:
            features_normalized = scaler.transform([task['data'][FEATURE_COLUMNS].values[0]])
        
        if 'lstm' in model_type.lower():
            features_reshaped = features_normalized.reshape((1, 1, features_normalized.shape[1]))
            pred_proba = cloud_model.predict(features_reshaped, verbose=0)
        elif 'cnn' in model_type.lower():
            features_reshaped = features_normalized.reshape((1, features_normalized.shape[1], 1))
            pred_proba = cloud_model.predict(features_reshaped, verbose=0)
        else:
            pred_proba = cloud_model.predict(features_normalized, verbose=0)
        
        pred = np.argmax(pred_proba[0])
        confidence = np.max(pred_proba[0])
        
        processing_time = task['complexity'] * 2 + random.uniform(1, 3)
        yield env.timeout(processing_time)
        
        total_latency = env.now - task['arrival']
        energy_consumed = task['complexity'] * 0.2 + network_latency * 0.1 + 0.3
        accuracy = 1.0 if pred == task['true_label'] else 0.0
        
        metrics['cloud']['tasks_processed'] += 1
        metrics['cloud']['latency'].append(total_latency)
        metrics['cloud']['energy'].append(energy_consumed)
        metrics['cloud']['accuracy'].append(accuracy)
        metrics['cloud']['processing_times'].append(processing_time)
        metrics['cloud']['total_cost'] += CLOUD_COST_PER_TASK
        
        if pred != 0:
            metrics['cloud']['faults_detected'] += 1
            metrics['total_faults_detected'] += 1
            logger.log_event(env.now, 'FAULT_DETECTED',
                           f"CLOUD detected complex fault: {FAULT_TYPES[pred]} "
                           f"(confidence: {confidence:.2f})")
        
        db_storage.append({
            'timestamp': env.now,
            'prediction': FAULT_TYPES[pred],
            'confidence': float(confidence),
            'cost': CLOUD_COST_PER_TASK
        })
        
        logger.log_task_completion(env.now, 'cloud', task['id'], total_latency,
                                 energy_consumed, accuracy, pred != 0)


def run_mqtt_simulation(broker_address='localhost', broker_port=1883,
                        topics=None, use_mqtt=True, simulation_time=1000,
                        sensor_publish_rate=0.5):
    """
    Run simulation with MQTT integration
    
    Args:
        broker_address: MQTT broker IP/hostname
        broker_port: MQTT broker port
        topics: MQTT topics to subscribe
        use_mqtt: True=use MQTT, False=synthetic only
        simulation_time: Simulation duration (minutes)
        sensor_publish_rate: How often sensors publish (seconds)
    """
    from main import (load_pretrained_models, create_metrics_dict, 
                      analyze_results)
    from dqn_scheduler import DQNScheduler
    from logger import SystemLogger
    from config import EDGE_CAPACITY, CLOUD_CAPACITY
    
    print("\n" + "="*80)
    print(" "*15 + "MQTT-ENABLED PREDICTIVE MAINTENANCE SIMULATION")
    print("="*80)
    
    mqtt_handler = None
    if use_mqtt:
        print("\n[Step 1/7] Initializing MQTT Handler...")
        mqtt_handler = MQTTSensorHandler(
            broker_address=broker_address,
            broker_port=broker_port,
            topics=topics or ["factory/sensors/#"],
            buffer_size=200
        )
        
        if not mqtt_handler.start():
            print("⚠ MQTT connection failed. Falling back to synthetic data.")
            use_mqtt = False
        else:
            print(f"✓ Connected! Waiting for sensor data...")
            time.sleep(2)
    else:
        print("\n[Step 1/7] MQTT disabled - using synthetic data only")
    
    print("\n[Step 2/7] Creating Intelligent MQTT-SimPy Bridge...")
    mqtt_bridge = MQTTSensorBridge(mqtt_handler, use_mqtt=use_mqtt)
    
    print("\n[Step 3/7] Loading Pre-trained Models...")
    edge_model, cloud_model, scaler, edge_cm, cloud_cm, models_loaded = load_pretrained_models()
    
    if not models_loaded:
        print("✗ Models not found. Run 'python train.py' first!")
        if mqtt_handler:
            mqtt_handler.stop()
        return None, None, None

    print("\n[Step 4/7] Training DQN Scheduler...")
    dqn_scheduler = DQNScheduler()
    dqn_scheduler.train(total_timesteps=1000)
    

    print("\n[Step 5/7] Setting up SimPy Environment...")
    metrics = create_metrics_dict()
    db_storage = []
    logger = SystemLogger(metrics)
    
    env = simpy.Environment()
    edge_resource = simpy.Resource(env, capacity=EDGE_CAPACITY)
    cloud_resource = simpy.Resource(env, capacity=CLOUD_CAPACITY)
    print(f"\n[Step 6/7] Starting Simulation ({simulation_time} minutes)...")
    print("-" * 80)
    if use_mqtt:
        print("📡 Using MQTT data with intelligent caching")
        print(f"   • Timeout reduced to 0.1s for fast fallback")
        print(f"   • Cache enabled for smooth operation")
    else:
        print("🔄 Generating synthetic data")
    
    env.process(mqtt_sensor_process(
        env, edge_resource, cloud_resource, dqn_scheduler, metrics, scaler,
        edge_model, cloud_model, logger, db_storage, 'lightgbm_lstm', mqtt_bridge
    ))
    
    try:
        env.run(until=simulation_time)
    except KeyboardInterrupt:
        print("\n⚠ Simulation interrupted by user")

    print(f"\n[Step 7/7] Simulation Complete! Analyzing results...")
    print("-" * 80)
    
    if mqtt_handler:
        mqtt_stats = mqtt_handler.get_statistics()
        bridge_stats = mqtt_bridge.get_statistics()
        
        print(f"\n📊 MQTT Statistics:")
        print(f"  • Messages Received: {mqtt_stats['messages_received']}")
        print(f"  • Messages Processed: {mqtt_stats['messages_processed']}")
        print(f"  • Connection Status: {'✓ Connected' if mqtt_stats['connected'] else '✗ Disconnected'}")
        
        print(f"\n📊 Data Source Statistics:")
        print(f"  • Direct MQTT readings: {bridge_stats['mqtt_readings']} ({bridge_stats['mqtt_percentage']:.1f}%)")
        print(f"  • Cached MQTT readings: {bridge_stats['cache_hits']} ({bridge_stats['cache_percentage']:.1f}%)")
        print(f"  • Synthetic fallback: {bridge_stats['synthetic_readings']}")
        print(f"  • Total timeouts: {bridge_stats['timeout_count']}")
        
        mqtt_handler.stop()
    
    if (metrics['edge']['tasks_processed'] + metrics['cloud']['tasks_processed']) > 0:
        analyze_results(metrics, edge_cm, cloud_cm, 'lightgbm_lstm_mqtt')
    else:
        print("⚠ No tasks were processed during simulation")
    
    return metrics, edge_cm, cloud_cm


if __name__ == "__main__":
    run_mqtt_simulation(
        broker_address='localhost',
        broker_port=1883,
        topics=['factory/sensors/#'],
        use_mqtt=True,
        simulation_time=1000,
        sensor_publish_rate=0.5
    )