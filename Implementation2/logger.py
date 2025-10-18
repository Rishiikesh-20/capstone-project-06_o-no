class SystemLogger:
    def __init__(self, metrics):
        self.metrics = metrics
        self.high_load_warned = False 

    def log_event(self, env_time, event_type, details):
        log_entry = {'timestamp': env_time, 'type': event_type, 'details': details}
        self.metrics['logs'].append(log_entry)
        if event_type in ['FAULT_DETECTED', 'SYSTEM_ALERT', 'OFFLOAD_DECISION']:
            print(f"[{env_time:.1f}] {event_type}: {details}")

    def check_and_alert_high_load(self, env_time, edge_load):
        if edge_load > 0.9 and not self.high_load_warned:
            self.log_event(env_time, 'SYSTEM_ALERT', 
                          f"⚠ Critical edge load detected: {edge_load*100:.1f}%")
            self.high_load_warned = True
        elif edge_load <= 0.7:
            self.high_load_warned = False  

    def log_task_completion(self, env_time, location, task_id, latency, energy, accuracy, fault_detected):
        self.metrics['timeline'].append({
            'time': env_time, 'task_id': task_id, 'location': location,
            'latency': latency, 'energy': energy, 'accuracy': accuracy, 'fault_detected': fault_detected
        })