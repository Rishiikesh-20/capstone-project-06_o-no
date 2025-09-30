import numpy as np

class EnhancedGA:
    def __init__(self, population_size=20, generations=10):
        self.population_size = population_size; self.generations = generations
    
    def fitness(self, chromosome, task_complexity, edge_load, cloud_queue, net_latency=5):
        if chromosome == 0:  
            latency = (task_complexity * 2) + (edge_load * 10 if edge_load > 0.8 else edge_load * 3)
            energy = task_complexity * 0.5; qos_penalty = 20 if edge_load > 0.9 else 0
        else: 
            latency = net_latency + (task_complexity * 1.5) + (cloud_queue * 0.5)
            energy = (task_complexity * 0.2) + 0.3; qos_penalty = 0
        return 0.4 * latency + 0.3 * energy + 0.3 * qos_penalty
    
    def schedule(self, task_complexity, edge_load, cloud_queue=0, net_latency=5):
        population = np.random.randint(0, 2, self.population_size)
        for _ in range(self.generations):
            fitness_scores = np.array([self.fitness(c, task_complexity, edge_load, cloud_queue, net_latency) for c in population])
            new_population = [population[idx1] if fitness_scores[idx1] < fitness_scores[idx2] else population[idx2] for idx1, idx2 in [np.random.choice(self.population_size, 2, replace=False) for _ in range(self.population_size)]]
            for i in range(0, self.population_size - 1, 2):
                if np.random.random() < 0.7: new_population[i], new_population[i+1] = new_population[i+1], new_population[i]
            for i in range(self.population_size):
                if np.random.random() < 0.1: new_population[i] = 1 - new_population[i]
            population = np.array(new_population)
        
        final_fitness = [self.fitness(c, task_complexity, edge_load, cloud_queue, net_latency) for c in population]
        decision = population[np.argmin(final_fitness)]
        min_fitness = np.min(final_fitness)
        return decision, min_fitness