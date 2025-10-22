import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    USE_SB3 = True
except ImportError:
    USE_SB3 = False
    print("⚠ Stable Baselines3 not available. Using simplified DQN implementation.")
class OffloadEnv(gym.Env):
    """Custom Gym Environment for task offloading decisions."""
    
    def __init__(self):
        super(OffloadEnv, self).__init__()
        
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([5.0, 1.0, 100.0, 50.0]),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(2)
        
        self.state = None
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        self.state = np.array([
            np.random.uniform(0.5, 3.0), 
            np.random.uniform(0.0, 1.0),   
            np.random.uniform(0, 20),     
            np.random.uniform(5, 15)      
        ], dtype=np.float32)
        return self.state, {}  
    
    def step(self, action):
        """Execute action and return reward."""
        task_complexity, edge_load, cloud_queue, net_latency = self.state
        
        if action == 0:  # Edge
            latency = (task_complexity * 2) + (edge_load * 10 if edge_load > 0.8 else edge_load * 3)
            energy = task_complexity * 0.5
            qos_penalty = 20 if edge_load > 0.9 else 0
        else:  # Cloud
            latency = net_latency + (task_complexity * 1.5) + (cloud_queue * 0.5)
            energy = (task_complexity * 0.2) + 0.3
            qos_penalty = 0
        
        cost = 0.4 * latency + 0.3 * energy + 0.3 * qos_penalty
        reward = -cost  
        
        self.state = np.array([
            np.random.uniform(0.5, 3.0),
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0, 20),
            np.random.uniform(5, 15)
        ], dtype=np.float32)
        
        terminated = False  
        truncated = False   
        info = {'cost': cost}
        
        return self.state, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        pass


class SimpleDQN:
    """Simplified DQN implementation using epsilon-greedy Q-learning."""
    
    def __init__(self, state_dim=4, action_dim=2, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.q_table = {}
        self.memory = deque(maxlen=2000)
    
    def _discretize_state(self, state):
        """Discretize continuous state for Q-table."""
        bins = [
            np.linspace(0, 5, 10),  
            np.linspace(0, 1, 10),   
            np.linspace(0, 100, 10),
            np.linspace(0, 50, 10)   
        ]
        discretized = tuple(np.digitize(s, b) for s, b in zip(state, bins))
        return discretized
    
    def predict(self, state):
        """Predict action for given state."""
        state_key = self._discretize_state(state)
        
        if random.random() < self.epsilon or state_key not in self.q_table:
            action = random.randint(0, self.action_dim - 1)
        else:
            action = np.argmax(self.q_table[state_key])
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self):
        """Learn from experiences."""
        if len(self.memory) < 32:
            return
        
        batch = random.sample(self.memory, 32)
        
        for state, action, reward, next_state, done in batch:
            state_key = self._discretize_state(state)
            next_state_key = self._discretize_state(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_dim)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_dim)
            
            target = reward
            if not done:
                target += self.gamma * np.max(self.q_table[next_state_key])
            
            self.q_table[state_key][action] += self.lr * (target - self.q_table[state_key][action])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path):
        """Save model (placeholder)."""
        pass
    
    def load(self, path):
        """Load model (placeholder)."""
        pass


class DQNScheduler:
    """DQN-based task scheduler for edge-cloud offloading."""
    
    def __init__(self, use_pretrained=False):
        self.env = OffloadEnv()
        self.use_sb3 = USE_SB3
        self.rewards_history = []
        
        if self.use_sb3 and not use_pretrained:
            self.model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=1e-3,
                buffer_size=10000,
                learning_starts=100,
                batch_size=32,
                tau=1.0,
                gamma=0.95,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=100,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                verbose=0
            )
        else:
            self.model = SimpleDQN(state_dim=4, action_dim=2)
    
    def train(self, total_timesteps=1000):
        print(f"\nTraining DQN Scheduler ({total_timesteps} timesteps)...")
        
        if self.use_sb3:
            self.model.learn(total_timesteps=total_timesteps)
        else:
            state, _ = self.env.reset()
            for step in range(total_timesteps):
                action = self.model.predict(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.model.remember(state, action, reward, next_state, done)
                self.model.learn()
                if done:
                    state, _ = self.env.reset()
                else:
                    state = next_state
                
                if step % 100 == 0:
                    test_rewards = []
                    for _ in range(10):
                        test_state, _ = self.env.reset()
                        _, test_reward, _, _, _ = self.env.step(self.model.predict(test_state))
                        test_rewards.append(test_reward)
                    self.rewards_history.append(np.mean(test_rewards))
        
        print("✓ DQN Scheduler training completed!")
    
    def schedule(self, task_complexity, edge_load, cloud_queue=0, net_latency=5):
        state = np.array([task_complexity, edge_load, cloud_queue, net_latency], dtype=np.float32)
        
        if self.use_sb3:
            action, _ = self.model.predict(state, deterministic=True)
            action = int(action)
        else:
            old_epsilon = self.model.epsilon
            self.model.epsilon = 0.0
            action = self.model.predict(state)
            self.model.epsilon = old_epsilon
        
        _, reward, _, _, info = self.env.step(action)
        cost = info.get('cost', 0)
        
        return action, -cost  

