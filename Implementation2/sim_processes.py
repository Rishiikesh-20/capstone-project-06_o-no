import random
from config import INTER_ARRIVAL_MEAN, FEATURE_COLUMNS, COMPLEXITY_THRESHOLD, FAULT_TYPES, EDGE_CAPACITY
from data_prep import generate_runtime_sensor_data
import numpy as np
def edge_process(env, task, edge_res, cloud_res, ga_scheduler, metrics, scaler, rf_model, dense_model, logger, db_storage):
    with edge_res.request() as req:
        yield req
        edge_load = min((len(edge_res.users) + len(edge_res.queue)) / EDGE_CAPACITY, 1.0)
        metrics['scheduling']['edge_load_history'].append((env.now, edge_load))
        
        features = task['data'][FEATURE_COLUMNS].values[0]
        features_normalized = scaler.transform([features])
        
        preprocess_time = 1.5 + random.uniform(0, 0.5); yield env.timeout(preprocess_time)
        
        decision, ga_fitness = ga_scheduler.schedule(task['complexity'], edge_load, len(cloud_res.queue), 5 + random.uniform(0, 2))
        metrics['scheduling']['ga_fitness_scores'].append(ga_fitness)
        metrics['scheduling']['offload_decisions'].append({'decision': 'edge' if decision == 0 else 'cloud'})
        
        if decision == 0 and task['complexity'] < COMPLEXITY_THRESHOLD:
            pred = rf_model.predict(features_normalized)[0]
            confidence = np.max(rf_model.predict_proba(features_normalized)[0])
            inference_time = 0.5 + task['complexity'] * 0.3; yield env.timeout(inference_time)
            
            total_latency = env.now - task['arrival']; energy_consumed = task['complexity'] * 0.5 + preprocess_time * 0.1
            accuracy = 1.0 if pred == task['true_label'] else 0.0
            
            metrics['edge']['tasks_processed'] += 1; metrics['edge']['latency'].append(total_latency)
            metrics['edge']['energy'].append(energy_consumed); metrics['edge']['accuracy'].append(accuracy)
            
            if pred != 0:
                metrics['edge']['faults_detected'] += 1; metrics['total_faults_detected'] += 1
                logger.log_event(env.now, 'FAULT_DETECTED', f"EDGE detected {FAULT_TYPES[pred]} (confidence: {confidence:.2f})")
            logger.log_task_completion(env.now, 'edge', task['id'], total_latency, energy_consumed, accuracy, pred != 0)
        else:
            logger.log_event(env.now, 'OFFLOAD_DECISION', f"Task {task['id']} offloaded to cloud (Complexity: {task['complexity']:.2f}, Edge Load: {edge_load:.2f})")
            task['preprocessed_features'] = features_normalized
            env.process(cloud_process(env, task, cloud_res, metrics, scaler, dense_model, logger, db_storage))

def cloud_process(env, task, cloud_res, metrics, scaler, dense_model, logger, db_storage):
    with cloud_res.request() as req:
        yield req
        network_latency = 5 + random.uniform(0, 3); yield env.timeout(network_latency)
        metrics['cloud']['network_latency'].append(network_latency)
        
        features_normalized = task.get('preprocessed_features')
        if features_normalized is None:
            features_normalized = scaler.transform([task['data'][FEATURE_COLUMNS].values[0]])
            
        pred_proba = dense_model.predict(features_normalized, verbose=0)
        pred = np.argmax(pred_proba[0]); confidence = np.max(pred_proba[0])
        processing_time = task['complexity'] * 2 + random.uniform(1, 3); yield env.timeout(processing_time)
        
        total_latency = env.now - task['arrival']; energy_consumed = task['complexity'] * 0.2 + network_latency * 0.1 + 0.3
        accuracy = 1.0 if pred == task['true_label'] else 0.0
        
        metrics['cloud']['tasks_processed'] += 1; metrics['cloud']['latency'].append(total_latency)
        metrics['cloud']['energy'].append(energy_consumed); metrics['cloud']['accuracy'].append(accuracy)
        
        if pred != 0:
            metrics['cloud']['faults_detected'] += 1; metrics['total_faults_detected'] += 1
            logger.log_event(env.now, 'FAULT_DETECTED', f"CLOUD detected complex fault: {FAULT_TYPES[pred]} (confidence: {confidence:.2f})")
        
        db_storage.append({'timestamp': env.now, 'prediction': FAULT_TYPES[pred], 'confidence': float(confidence)})
        logger.log_task_completion(env.now, 'cloud', task['id'], total_latency, energy_consumed, accuracy, pred != 0)

def sensor_process(env, edge_res, cloud_res, ga_scheduler, metrics, scaler, rf_model, dense_model, logger, db_storage):
    scenarios = list(FAULT_TYPES.values()); scenario_probs = [0.7, 0.05, 0.05, 0.05, 0.05, 0.1]
    task_id = 0
    while True:
        yield env.timeout(random.expovariate(1/INTER_ARRIVAL_MEAN))
        scenario = np.random.choice(scenarios, p=scenario_probs)
        data = generate_runtime_sensor_data(scenario)
        task = {'id': task_id, 'data': data, 'arrival': env.now, 'complexity': random.uniform(0.5, 3), 'true_label': data['Fault_Label'].iloc[0]}
        task_id += 1; metrics['total_tasks'] += 1
        env.process(edge_process(env, task, edge_res, cloud_res, ga_scheduler, metrics, scaler, rf_model, dense_model, logger, db_storage))