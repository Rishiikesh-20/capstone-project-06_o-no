import random
import numpy as np
from config import (INTER_ARRIVAL_MEAN, FEATURE_COLUMNS, COMPLEXITY_THRESHOLD, 
                    FAULT_TYPES, EDGE_CAPACITY, NET_LATENCY_MEAN, NET_LATENCY_STD,
                    CLOUD_COST_PER_TASK)
from data_prep import generate_runtime_sensor_data


def edge_process(env, task, edge_res, cloud_res, dqn_scheduler, metrics, scaler, 
                 edge_model, cloud_model, logger, db_storage, model_type='lightgbm_lstm'):
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


def sensor_process(env, edge_res, cloud_res, dqn_scheduler, metrics, scaler, 
                   edge_model, cloud_model, logger, db_storage, model_type='lightgbm_lstm',
                   fault_prob_scale=1.0):
    scenarios = list(FAULT_TYPES.values())
    
    # Adjusted probabilities to better reflect training data distribution
    # This matches the augmented training data more closely for realistic simulation
    base_probs = [0.78, 0.04, 0.01, 0.08, 0.07, 0.02]
    # Normal: 78%, Tool Wear: 4%, Heat Dissipation: 1%, Power: 8%, Overstrain: 7%, Random: 2%
    
    if fault_prob_scale != 1.0:
        fault_probs = [p * fault_prob_scale for p in base_probs[1:]]
        normal_prob = max(0.1, 1.0 - sum(fault_probs)) 
        
        total = normal_prob + sum(fault_probs)
        scenario_probs = [normal_prob / total] + [fp / total for fp in fault_probs]
    else:
        scenario_probs = base_probs
    
    task_id = 0
    while True:
        yield env.timeout(random.expovariate(1/INTER_ARRIVAL_MEAN))
        
        scenario = np.random.choice(scenarios, p=scenario_probs)
        data = generate_runtime_sensor_data(scenario)
        
        task = {
            'id': task_id,
            'data': data,
            'arrival': env.now,
            'complexity': random.uniform(0.5, 3),
            'true_label': data['Fault_Label'].iloc[0]
        }
        
        task_id += 1
        metrics['total_tasks'] += 1
        
        env.process(edge_process(env, task, edge_res, cloud_res, dqn_scheduler, 
                                metrics, scaler, edge_model, cloud_model, logger, 
                                db_storage, model_type))