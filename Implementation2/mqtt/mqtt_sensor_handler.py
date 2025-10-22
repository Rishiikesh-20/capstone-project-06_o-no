"""
MQTT Sensor Handler for Predictive Maintenance System
Bridges real-time MQTT sensor data with SimPy simulation
"""

import paho.mqtt.client as mqtt
import json
import threading
import queue
import time
from collections import deque
import numpy as np


class MQTTSensorHandler:
    """
    Handles MQTT communication for industrial sensors.
    Receives real-time sensor data and makes it available to SimPy simulation.
    """
    
    def __init__(self, broker_address="localhost", broker_port=1883, 
                 topics=None, buffer_size=100):
        """
        Initialize MQTT Sensor Handler
        
        Args:
            broker_address: MQTT broker IP/hostname (default: localhost)
            broker_port: MQTT broker port (default: 1883)
            topics: List of MQTT topics to subscribe to
            buffer_size: Size of sensor data buffer
        """
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.topics = topics or ["factory/sensors/#"]
        self.data_queue = queue.Queue(maxsize=buffer_size)
        self.sensor_buffer = deque(maxlen=buffer_size)
        self.connected = False
        self.running = False
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'connection_errors': 0,
            'last_message_time': None
        }
        self.client = mqtt.Client(client_id="predictive_maintenance_system")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        print(f"✓ MQTT Handler initialized (Broker: {broker_address}:{broker_port})")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        if rc == 0:
            self.connected = True
            print(f"✓ Connected to MQTT broker: {self.broker_address}:{self.broker_port}")
            for topic in self.topics:
                client.subscribe(topic)
                print(f"  • Subscribed to: {topic}")
        else:
            self.connected = False
            self.stats['connection_errors'] += 1
            error_messages = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorized"
            }
            print(f"✗ Connection failed: {error_messages.get(rc, f'Unknown error {rc}')}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker"""
        self.connected = False
        if rc != 0:
            print(f"⚠ Unexpected disconnection from MQTT broker (code: {rc})")
            print("  Attempting to reconnect...")
    
    def _on_message(self, client, userdata, msg):
        """Callback when a message is received"""
        try:
            payload = json.loads(msg.payload.decode())
            
           
            sensor_data = {
                'topic': msg.topic,
                'timestamp': time.time(),
                'sensor_id': self._extract_sensor_id(msg.topic),
                'data': payload
            }
            
            
            if not self.data_queue.full():
                self.data_queue.put(sensor_data)
                self.sensor_buffer.append(sensor_data)
                self.stats['messages_received'] += 1
                self.stats['last_message_time'] = time.time()
            else:
                
                try:
                    self.data_queue.get_nowait() 
                    self.data_queue.put(sensor_data)
                except queue.Empty:
                    pass
                    
        except json.JSONDecodeError:
            print(f"⚠ Invalid JSON from topic {msg.topic}: {msg.payload}")
        except Exception as e:
            print(f"✗ Error processing message: {e}")
    
    def _extract_sensor_id(self, topic):
        """Extract sensor ID from MQTT topic"""
        parts = topic.split('/')
        return parts[2] if len(parts) > 2 else "unknown"
    
    def start(self):
        """Start MQTT client in background thread"""
        if self.running:
            print("⚠ MQTT Handler already running")
            return
        
        try:
            self.running = True
            self.client.connect(self.broker_address, self.broker_port, keepalive=60)
            
            
            self.client.loop_start()
            
            print("✓ MQTT Handler started successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to start MQTT Handler: {e}")
            self.running = False
            return False
    
    def stop(self):
        """Stop MQTT client"""
        if not self.running:
            return
        
        print("\nStopping MQTT Handler...")
        self.running = False
        self.client.loop_stop()
        self.client.disconnect()
        print("✓ MQTT Handler stopped")
    
    def get_sensor_data(self, timeout=0.1):
        """
        Get next sensor reading from queue (non-blocking)
        
        Args:
            timeout: Maximum time to wait for data (seconds)
            
        Returns:
            Sensor data dict or None if no data available
        """
        try:
            data = self.data_queue.get(timeout=timeout)
            self.stats['messages_processed'] += 1
            return data
        except queue.Empty:
            return None
    
    def get_latest_readings(self, sensor_id=None, count=10):
        """
        Get recent sensor readings from buffer
        
        Args:
            sensor_id: Filter by specific sensor (None = all sensors)
            count: Number of recent readings to return
            
        Returns:
            List of sensor readings
        """
        readings = list(self.sensor_buffer)
        
        if sensor_id:
            readings = [r for r in readings if r['sensor_id'] == sensor_id]
        
        return readings[-count:]
    
    def get_statistics(self):
        """Get handler statistics"""
        return {
            **self.stats,
            'connected': self.connected,
            'queue_size': self.data_queue.qsize(),
            'buffer_size': len(self.sensor_buffer)
        }
    
    def publish_alert(self, alert_type, message, sensor_id=None):
        """
        Publish alert/prediction back to MQTT broker
        
        Args:
            alert_type: Type of alert (e.g., 'FAULT_DETECTED', 'MAINTENANCE_REQUIRED')
            message: Alert message
            sensor_id: Related sensor ID (optional)
        """
        if not self.connected:
            print("⚠ Cannot publish - not connected to broker")
            return False
        
        alert_data = {
            'timestamp': time.time(),
            'alert_type': alert_type,
            'message': message,
            'sensor_id': sensor_id
        }
        
        topic = f"factory/alerts/{sensor_id}" if sensor_id else "factory/alerts/general"
        
        try:
            self.client.publish(topic, json.dumps(alert_data), qos=1)
            return True
        except Exception as e:
            print(f"✗ Failed to publish alert: {e}")
            return False


class MQTTSensorSimulator:
    """
    Simulates industrial sensors publishing to MQTT
    Now with faster publishing rate for better simulation
    """
    
    def __init__(self, broker_address="localhost", broker_port=1883, num_sensors=5):
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.num_sensors = num_sensors
        self.running = False
        
        self.client = mqtt.Client(client_id="sensor_simulator")
        
        
        self.sensors = []
        sensor_types = ['temperature', 'vibration', 'pressure', 'rpm', 'torque', 'tool_wear']
        
        for i in range(num_sensors):
            sensor_type = sensor_types[i % len(sensor_types)]
            
            
            if sensor_type == 'temperature':
                base_value = np.random.uniform(295, 305) 
                noise_level = 2.0
            elif sensor_type == 'vibration':
                base_value = np.random.uniform(0.3, 1.0)
                noise_level = 0.1
            elif sensor_type == 'pressure':
                base_value = np.random.uniform(95, 105)
                noise_level = 2.0
            elif sensor_type == 'rpm':
                base_value = np.random.uniform(1400, 1600)
                noise_level = 50.0
            elif sensor_type == 'torque':
                base_value = np.random.uniform(35, 45)
                noise_level = 3.0
            else:  
                base_value = np.random.uniform(80, 120)
                noise_level = 5.0
            
            self.sensors.append({
                'id': f'machine{i}',
                'type': sensor_type,
                'base_value': base_value,
                'noise_level': noise_level,
                'fault_probability': 0.15 
            })
        
        print(f"✓ Sensor Simulator initialized ({num_sensors} virtual sensors)")
        for sensor in self.sensors:
            print(f"  • {sensor['id']}: {sensor['type']} (base: {sensor['base_value']:.1f})")
    
    def start(self, publish_interval=0.5):
        """Start publishing simulated sensor data"""
        try:
            self.client.connect(self.broker_address, self.broker_port)
            self.client.loop_start()
            self.running = True
            
            print(f"✓ Sensor Simulator started (Publishing every {publish_interval}s)")
           
            publish_count = 0
            while self.running:
                for sensor in self.sensors:
                    
                    value = sensor['base_value'] + np.random.normal(0, sensor['noise_level'])
                    
                    
                    if np.random.random() < sensor['fault_probability']:
                      
                        if sensor['type'] == 'temperature':
                            value += np.random.uniform(10, 20)  
                        elif sensor['type'] == 'vibration':
                            value *= np.random.uniform(2.0, 3.0)  
                        elif sensor['type'] == 'rpm':
                            value -= np.random.uniform(200, 400)  
                        elif sensor['type'] == 'torque':
                            value *= np.random.uniform(1.5, 2.0)  
                        elif sensor['type'] == 'tool_wear':
                            value += np.random.uniform(100, 150)  
                    
                    if sensor['type'] == 'tool_wear':
                        value = max(0, min(value, 250))
                    elif sensor['type'] == 'rpm':
                        value = max(0, value)
                    
                    payload = {
                        'sensor_type': sensor['type'],
                        'value': round(value, 2),
                        'unit': self._get_unit(sensor['type']),
                        'timestamp': time.time(),
                        'sensor_id': sensor['id']
                    }
                    
                    topic = f"factory/sensors/{sensor['id']}/{sensor['type']}"
                    self.client.publish(topic, json.dumps(payload))
                
                publish_count += 1
                if publish_count % 20 == 0: 
                    print(f"  📡 Published {publish_count * self.num_sensors} messages...")
                
                time.sleep(publish_interval)
                
        except KeyboardInterrupt:
            print("\n⚠ Simulator stopped by user")
        except Exception as e:
            print(f"✗ Simulator error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop simulator"""
        self.running = False
        self.client.loop_stop()
        self.client.disconnect()
        print("✓ Sensor Simulator stopped")
    
    def _get_unit(self, sensor_type):
        """Get measurement unit for sensor type"""
        units = {
            'temperature': '°C',
            'vibration': 'mm/s',
            'pressure': 'bar',
            'rpm': 'RPM',
            'torque': 'Nm',
            'tool_wear': 'min'
        }
        return units.get(sensor_type, 'units')

if __name__ == "__main__":
    print("="*70)
    print("MQTT Sensor Handler - Test Mode")
    print("="*70)
    
    print("\n[Test 1] Starting sensor simulator...")
    simulator = MQTTSensorSimulator(num_sensors=3)
    simulator_thread = threading.Thread(target=simulator.start, args=(2.0,))
    simulator_thread.daemon = True
    simulator_thread.start()
    
    time.sleep(2)

    print("\n[Test 2] Starting MQTT handler...")
    handler = MQTTSensorHandler(topics=["factory/sensors/#"])
    handler.start()
    
    time.sleep(2)  
    
    print("\n[Test 3] Receiving sensor data (10 seconds)...")
    start_time = time.time()
    while time.time() - start_time < 10:
        data = handler.get_sensor_data(timeout=1.0)
        if data:
            print(f"  • {data['sensor_id']}: {data['data']['sensor_type']} = "
                  f"{data['data']['value']} {data['data']['unit']}")
    
    print("\n[Test 4] Statistics:")
    stats = handler.get_statistics()
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    print("\n[Test 5] Publishing test alert...")
    handler.publish_alert("TEST_ALERT", "System test completed", "machine0")
    
    print("\n[Cleanup] Stopping services...")
    simulator.stop()
    handler.stop()
    
    print("\n✓ All tests completed!")