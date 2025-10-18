RANDOM_SEED = 42
SIMULATION_TIME = 5000 
INTER_ARRIVAL_MEAN = 10  
EDGE_CAPACITY = 5 
CLOUD_CAPACITY = 100 
NUM_FAULT_CLASSES = 6  
COMPLEXITY_THRESHOLD = 1.5  
DATASET_PATH = 'dataset.csv' 

NET_LATENCY_MEAN = 10  
NET_LATENCY_STD = 5    

CLOUD_COST_PER_TASK = 0.01  

FEATURE_COLUMNS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]


FAULT_TYPES = {
    0: 'Normal',
    1: 'Tool Wear Failure',
    2: 'Heat Dissipation Failure',
    3: 'Power Failure',
    4: 'Overstrain Failure',
    5: 'Random Failure'
}
