#!/usr/bin/env python3
"""Quick MQTT System Test"""

import time
import threading
from mqtt_sensor_handler import MQTTSensorHandler, MQTTSensorSimulator

def main():
    print("="*70)
    print(" "*20 + "MQTT SYSTEM TEST")
    print("="*70)
    
    # Test 1: Start simulator
    print("\n[1/3] Starting sensor simulator...")
    simulator = MQTTSensorSimulator(num_sensors=3)
    sim_thread = threading.Thread(target=simulator.start, args=(2.0,))
    sim_thread.daemon = True
    sim_thread.start()
    time.sleep(2)
    
    # Test 2: Connect handler
    print("\n[2/3] Connecting to MQTT broker...")
    handler = MQTTSensorHandler(
        broker_address="localhost",
        broker_port=1883,
        topics=["factory/sensors/#"]
    )
    
    if not handler.start():
        print("✗ Connection failed! Check if mosquitto is running:")
        print("  sudo systemctl start mosquitto")
        return
    
    time.sleep(2)
    
    # Test 3: Receive data
    print("\n[3/3] Receiving sensor data (10 seconds)...\n")
    start_time = time.time()
    count = 0
    
    while time.time() - start_time < 10:
        data = handler.get_sensor_data(timeout=1.0)
        if data:
            count += 1
            print(f"  [{count}] {data['sensor_id']}: "
                  f"{data['data']['sensor_type']} = "
                  f"{data['data']['value']:.2f} {data['data']['unit']}")
    
    # Results
    stats = handler.get_statistics()
    print(f"\n📊 Results:")
    print(f"  • Messages Received: {stats['messages_received']}")
    print(f"  • Status: {'✓ SUCCESS' if count > 0 else '✗ FAILED'}")
    
    # Cleanup
    simulator.stop()
    handler.stop()
    print("\n✓ Test complete!")

if __name__ == "__main__":
    main()