
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config import FEATURE_COLUMNS, FAULT_TYPES, NUM_FAULT_CLASSES
from data_prep import generate_runtime_sensor_data
from dqn_scheduler import DQNScheduler, OffloadEnv


class TestDataGeneration:
    def test_generate_normal_scenario(self):
        data = generate_runtime_sensor_data('Normal')
        
        assert isinstance(data, pd.DataFrame), "Should return DataFrame"
        assert len(data) == 1, "Should return single sample"
        assert all(col in data.columns for col in FEATURE_COLUMNS), "Should have all feature columns"
        assert 'Fault_Label' in data.columns, "Should have Fault_Label"
        assert data['Fault_Label'].iloc[0] == 0, "Normal scenario should have label 0"
    
    def test_generate_fault_scenarios(self):
        fault_scenarios = ['Tool Wear Failure', 'Heat Dissipation Failure', 
                          'Power Failure', 'Overstrain Failure']
        
        for scenario in fault_scenarios:
            data = generate_runtime_sensor_data(scenario)
            
            assert isinstance(data, pd.DataFrame), f"Should return DataFrame for {scenario}"
            assert data['Fault_Label'].iloc[0] != 0, f"{scenario} should have non-zero label"
            assert data['Tool wear [min]'].iloc[0] >= 0, "Tool wear should be non-negative"
    
    def test_sensor_data_ranges(self):
        data = generate_runtime_sensor_data('Normal')
        
        assert 250 < data['Air temperature [K]'].iloc[0] < 350, "Air temp should be reasonable"
        assert 250 < data['Process temperature [K]'].iloc[0] < 400, "Process temp should be reasonable"
        assert 500 < data['Rotational speed [rpm]'].iloc[0] < 3000, "RPM should be reasonable"
        assert 0 < data['Torque [Nm]'].iloc[0] < 200, "Torque should be reasonable"
        assert data['Tool wear [min]'].iloc[0] >= 0, "Tool wear should be non-negative"


class TestDQNScheduler:
    
    @pytest.fixture
    def scheduler(self):
        return DQNScheduler()
    
    @pytest.fixture
    def env(self):
        return OffloadEnv()
    
    def test_scheduler_initialization(self, scheduler):
        assert scheduler is not None, "Scheduler should be created"
        assert hasattr(scheduler, 'env'), "Should have environment"
        assert hasattr(scheduler, 'model'), "Should have model"
    
    def test_environment_reset(self, env):
        state = env.reset()
        
        assert isinstance(state, np.ndarray), "State should be numpy array"
        assert len(state) == 4, "State should have 4 dimensions"
        assert all(0 <= s < 1000 for s in state), "State values should be in reasonable range"
    
    def test_environment_step(self, env):
        env.reset()
        
        for action in [0, 1]:
            next_state, reward, done, info = env.step(action)
            
            assert isinstance(next_state, np.ndarray), "Next state should be numpy array"
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            assert isinstance(done, bool), "Done should be boolean"
            assert 'cost' in info, "Info should contain cost"
    
    def test_scheduler_decision(self, scheduler):
        decision, cost = scheduler.schedule(
            task_complexity=1.5,
            edge_load=0.5,
            cloud_queue=5,
            net_latency=10
        )
        
        assert decision in [0, 1], "Decision should be 0 (edge) or 1 (cloud)"
        assert isinstance(cost, (int, float)), "Cost should be numeric"
    
    def test_scheduler_with_different_loads(self, scheduler):
        decisions_high_load = []
        for _ in range(10):
            decision, _ = scheduler.schedule(2.0, 0.95, 5, 10)
            decisions_high_load.append(decision)
        
        decisions_low_load = []
        for _ in range(10):
            decision, _ = scheduler.schedule(0.8, 0.1, 5, 10)
            decisions_low_load.append(decision)
        
        assert len(decisions_high_load) == 10, "Should make 10 decisions"
        assert len(decisions_low_load) == 10, "Should make 10 decisions"


class TestModelTraining:
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n_samples = 100
        
        X = np.random.randn(n_samples, len(FEATURE_COLUMNS))
        y = np.random.randint(0, NUM_FAULT_CLASSES, n_samples)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, scaler
    
    def test_data_scaling(self, sample_data):
        X_scaled, y, scaler = sample_data
        
        assert X_scaled.shape[0] == 100, "Should have 100 samples"
        assert X_scaled.shape[1] == len(FEATURE_COLUMNS), "Should have correct feature count"
        
        assert np.abs(X_scaled.mean()) < 0.2, "Scaled data should have ~zero mean"
        assert np.abs(X_scaled.std() - 1.0) < 0.2, "Scaled data should have ~unit variance"
    
    def test_fault_types_mapping(self):
        assert len(FAULT_TYPES) == NUM_FAULT_CLASSES, "Should have correct number of fault types"
        assert 0 in FAULT_TYPES, "Should include Normal (0)"
        assert FAULT_TYPES[0] == 'Normal', "Class 0 should be Normal"


class TestMetricsAndLogging:
    
    def test_metrics_structure(self):
        from main import create_metrics_dict
        
        metrics = create_metrics_dict()
        
        assert 'total_tasks' in metrics, "Should have total_tasks"
        assert 'edge' in metrics, "Should have edge metrics"
        assert 'cloud' in metrics, "Should have cloud metrics"
        assert 'scheduling' in metrics, "Should have scheduling metrics"
        
        assert 'tasks_processed' in metrics['edge'], "Edge should track tasks_processed"
        assert 'latency' in metrics['edge'], "Edge should track latency"
        assert 'accuracy' in metrics['edge'], "Edge should track accuracy"
        
        assert 'total_cost' in metrics['cloud'], "Cloud should track total_cost"
        assert 'network_latency' in metrics['cloud'], "Cloud should track network_latency"
    
    def test_logger_initialization(self):
        from logger import SystemLogger
        from main import create_metrics_dict
        
        metrics = create_metrics_dict()
        logger = SystemLogger(metrics)
        
        assert logger is not None, "Logger should be created"
        assert logger.metrics is metrics, "Logger should reference metrics"
        assert hasattr(logger, 'high_load_warned'), "Logger should have high_load_warned flag"
    
    def test_logger_event_logging(self):
        from logger import SystemLogger
        from main import create_metrics_dict
        
        metrics = create_metrics_dict()
        logger = SystemLogger(metrics)
        
        logger.log_event(10.5, 'TEST_EVENT', 'Test details')
        
        assert len(metrics['logs']) == 1, "Should have logged one event"
        assert metrics['logs'][0]['timestamp'] == 10.5, "Should record correct timestamp"
        assert metrics['logs'][0]['type'] == 'TEST_EVENT', "Should record correct type"


class TestConfiguration:
    
    def test_config_constants(self):
        from config import (RANDOM_SEED, SIMULATION_TIME, EDGE_CAPACITY, 
                           CLOUD_CAPACITY, NET_LATENCY_MEAN, NET_LATENCY_STD)
        
        assert isinstance(RANDOM_SEED, int), "RANDOM_SEED should be int"
        assert isinstance(SIMULATION_TIME, int), "SIMULATION_TIME should be int"
        assert EDGE_CAPACITY > 0, "EDGE_CAPACITY should be positive"
        assert CLOUD_CAPACITY > EDGE_CAPACITY, "CLOUD_CAPACITY should be greater than EDGE"
        assert NET_LATENCY_MEAN > 0, "NET_LATENCY_MEAN should be positive"
        assert NET_LATENCY_STD >= 0, "NET_LATENCY_STD should be non-negative"
    
    def test_feature_columns(self):
        assert len(FEATURE_COLUMNS) == 5, "Should have 5 feature columns"
        assert all(isinstance(col, str) for col in FEATURE_COLUMNS), "All columns should be strings"


class TestIntegration:
    
    @pytest.mark.slow
    def test_single_task_flow(self):
        pass
    
    def test_cost_calculation(self):
        from config import CLOUD_COST_PER_TASK
        
        num_cloud_tasks = 100
        expected_cost = num_cloud_tasks * CLOUD_COST_PER_TASK
        
        assert expected_cost == 100 * CLOUD_COST_PER_TASK, "Cost should scale linearly"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])

