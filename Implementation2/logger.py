class SystemLogger:
    def __init__(self, metrics):
        self.metrics = metrics

    def log_event(self, env_time, event_type, details):
        log_entry = {'timestamp': env_time, 'type': event_type, 'details': details}
        self.metrics['logs'].append(log_entry)
        if event_type in ['FAULT_DETECTED', 'SYSTEM_ALERT', 'OFFLOAD_DECISION']:
            print(f"[{env_time:.1f}] {event_type}: {details}")

    def log_task_completion(self, env_time, location, task_id, latency, energy, accuracy, fault_detected):
        self.metrics['timeline'].append({
            'time': env_time, 'task_id': task_id, 'location': location,
            'latency': latency, 'energy': energy, 'accuracy': accuracy, 'fault_detected': fault_detected
        })