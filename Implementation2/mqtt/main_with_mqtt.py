"""
Complete Main Script with MQTT Integration
Run this directly with: python main_with_mqtt.py
"""

import threading
import time
import sys

def start_sensor_simulator(publish_rate=0.5, num_sensors=5):
    """Start MQTT sensor simulator in background"""
    from mqtt_sensor_handler import MQTTSensorSimulator
    
    print(f"\n🔧 Starting sensor simulator...")
    print(f"   • Number of sensors: {num_sensors}")
    print(f"   • Publish rate: {publish_rate}s")
    
    simulator = MQTTSensorSimulator(num_sensors=num_sensors)
    sim_thread = threading.Thread(target=simulator.start, args=(publish_rate,))
    sim_thread.daemon = True
    sim_thread.start()
    
    time.sleep(2) 
    return simulator, sim_thread


def run_with_mqtt_simulator(simulation_time=1000, num_sensors=5, publish_rate=0.5):
    """
    Run simulation with built-in MQTT simulator
    Best for testing without real hardware
    """
    from mqtt_integration import run_mqtt_simulation
    
    print("\n" + "="*80)
    print(" "*20 + "MODE: MQTT WITH SIMULATOR")
    print("="*80)
    
    simulator, _ = start_sensor_simulator(publish_rate, num_sensors)
    
    try:
        metrics, edge_cm, cloud_cm = run_mqtt_simulation(
            broker_address='localhost',
            broker_port=1883,
            topics=['factory/sensors/#'],
            use_mqtt=True,
            simulation_time=simulation_time
        )
        
        return metrics, edge_cm, cloud_cm
        
    finally:
        simulator.stop()


def run_with_real_mqtt(broker_address='localhost', broker_port=1883,
                       simulation_time=1000):
    """
    Run simulation with real MQTT sensors
    Use this when you have actual hardware publishing data
    """
    from mqtt_integration import run_mqtt_simulation
    
    print("\n" + "="*80)
    print(" "*20 + "MODE: REAL MQTT SENSORS")
    print("="*80)
    print(f"\nConnecting to broker: {broker_address}:{broker_port}")
    print("Make sure your sensors are publishing to factory/sensors/# topics\n")
    
    metrics, edge_cm, cloud_cm = run_mqtt_simulation(
        broker_address=broker_address,
        broker_port=broker_port,
        topics=['factory/sensors/#'],
        use_mqtt=True,
        simulation_time=simulation_time
    )
    
    return metrics, edge_cm, cloud_cm


def run_synthetic_only(simulation_time=1000):
    """
    Run simulation without MQTT (pure synthetic data)
    Fastest mode for testing
    """
    from main import run_simulation, analyze_results
    
    print("\n" + "="*80)
    print(" "*20 + "MODE: SYNTHETIC DATA ONLY")
    print("="*80)
    
    metrics, edge_cm, cloud_cm, _, _, _ = run_simulation(
        model_type='lightgbm_lstm',
        simulation_time=simulation_time
    )
    
    if (metrics['edge']['tasks_processed'] + metrics['cloud']['tasks_processed']) > 0:
        analyze_results(metrics, edge_cm, cloud_cm, 'lightgbm_lstm')
    
    return metrics, edge_cm, cloud_cm


def main():
    """Main entry point with mode selection"""
    
    print("\n" + "="*80)
    print(" "*15 + "PREDICTIVE MAINTENANCE SIMULATION SYSTEM")
    print("="*80)
    
    print("\nSelect simulation mode:")
    print("  1. MQTT with Simulator (Recommended - no hardware needed)")
    print("  2. Real MQTT Sensors (requires hardware)")
    print("  3. Synthetic Data Only (no MQTT)")
    print("  4. Quick Test (100 minutes)")
    
    try:
        choice = input("\nEnter choice (1-4) [default=1]: ").strip() or "1"
        
        if choice == "1":
            print("\n✓ Selected: MQTT with Simulator")
            
            sim_time = input("Simulation duration (minutes) [default=1000]: ").strip()
            sim_time = int(sim_time) if sim_time else 1000
            
            num_sensors = input("Number of virtual sensors [default=5]: ").strip()
            num_sensors = int(num_sensors) if num_sensors else 5
            
            publish_rate = input("Sensor publish rate (seconds) [default=0.5]: ").strip()
            publish_rate = float(publish_rate) if publish_rate else 0.5
            
            metrics, edge_cm, cloud_cm = run_with_mqtt_simulator(
                simulation_time=sim_time,
                num_sensors=num_sensors,
                publish_rate=publish_rate
            )
            
        elif choice == "2":
            print("\n✓ Selected: Real MQTT Sensors")
            
            broker = input("MQTT broker address [default=localhost]: ").strip() or "localhost"
            port = input("MQTT broker port [default=1883]: ").strip()
            port = int(port) if port else 1883
            
            sim_time = input("Simulation duration (minutes) [default=1000]: ").strip()
            sim_time = int(sim_time) if sim_time else 1000
            
            metrics, edge_cm, cloud_cm = run_with_real_mqtt(
                broker_address=broker,
                broker_port=port,
                simulation_time=sim_time
            )
            
        elif choice == "3":
            print("\n✓ Selected: Synthetic Data Only")
            
            sim_time = input("Simulation duration (minutes) [default=1000]: ").strip()
            sim_time = int(sim_time) if sim_time else 1000
            
            metrics, edge_cm, cloud_cm = run_synthetic_only(simulation_time=sim_time)
            
        elif choice == "4":
            print("\n✓ Selected: Quick Test Mode (100 minutes)")
            metrics, edge_cm, cloud_cm = run_with_mqtt_simulator(
                simulation_time=100,
                num_sensors=3,
                publish_rate=0.5
            )
            
        else:
            print(f"Invalid choice: {choice}")
            return
        
        if metrics and (metrics['edge']['tasks_processed'] + metrics['cloud']['tasks_processed']) > 0:
            print("\n" + "="*80)
            print(" "*25 + "SIMULATION COMPLETE!")
            print("="*80)
            print(f"\n Total tasks processed: {metrics['edge']['tasks_processed'] + metrics['cloud']['tasks_processed']}")
            print(f" Faults detected: {metrics['total_faults_detected']}")
            print(f"System working correctly!")
        
    except KeyboardInterrupt:
        print("\n\n  Simulation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import os
    model_files = [
        'edge_lightgbm_model.pkl',
        'cloud_lstm_model.keras',
        'scaler.pkl'
    ]
    
    missing_models = [f for f in model_files if not os.path.exists(f)]
    
    if missing_models:
        print("\n⚠️  WARNING: Pre-trained models not found!")
        print("Missing files:")
        for f in missing_models:
            print(f"  • {f}")
        print("\n Action required:")
        print("  Run 'python train.py' first to train the models")
        print("\nDo you want to continue anyway? (simulation may fail)")
        response = input("Continue? (yes/no) [no]: ").strip().lower()
        if response not in ['yes', 'y']:
            sys.exit(0)
    
    main()