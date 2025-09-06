"""
Real-Time Baseline Trainer for Panic Attack Detection System
===========================================================

This script reads real-time sensor data from Arduino and creates a personalized
baseline for the user. The baseline is then saved to medical_baselines.pkl
for use with the panic attack detection system.
"""

import time
import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime, timedelta
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

class RealTimeBaselineTrainer:    
    def __init__(self, port='COM3', baudrate=9600, user_id=None):
        self.port = port
        self.baudrate = baudrate
        self.user_id = user_id or f"USER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.baseline_duration = 300 
        self.sampling_rate = 10 
        self.total_samples = self.baseline_duration * self.sampling_rate
        
        self.sensor_data = {
            'timestamp': [],
            'heart_rate': [],
            'eda': [],
            'respiration': [],
            'temperature': [],
            'tremor': []
        }
        self.serial_conn = None
        self.data_queue = queue.Queue()
        self.collecting = False
        self.baseline = None
        print(f"🎯 Real-Time Baseline Trainer Initialized")
        print(f"   👤 User ID: {self.user_id}")
        print(f"   ⏱️  Duration: {self.baseline_duration} seconds ({self.baseline_duration//60} minutes)")
        print(f"   📊 Sampling Rate: {self.sampling_rate} Hz")
        print(f"   📈 Total Samples: {self.total_samples}")
    
    def connect_arduino(self):
        try:
            print(f"🔌 Connecting to Arduino on {self.port}...")
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            time.sleep(2)  # Wait for Arduino to initialize
            print(f"✅ Arduino connected successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Arduino: {e}")
            print(f"💡 Make sure Arduino is connected to {self.port}")
            return False
    
    def disconnect_arduino(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("🔌 Arduino disconnected")
    
    def read_sensor_data(self):
        while self.collecting:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line:
                        # Parse sensor data (format: HR,EDA,RESP,TEMP,ACC_X,ACC_Y,ACC_Z)
                        data = self.parse_sensor_data(line)
                        if data:
                            self.data_queue.put(data)
            except Exception as e:
                print(f"⚠️ Error reading sensor data: {e}")
            time.sleep(0.1)  # 10 Hz sampling rate
    
    def parse_sensor_data(self, line):
        try:
            parts = line.split(',')
            if len(parts) >= 7:
                return {
                    'timestamp': time.time(),
                    'heart_rate': float(parts[0]),
                    'eda': float(parts[1]),
                    'respiration': float(parts[2]),
                    'temperature': float(parts[3]),
                    'acc_x': float(parts[4]),
                    'acc_y': float(parts[5]),
                    'acc_z': float(parts[6])
                }
        except (ValueError, IndexError) as e:
            print(f"⚠️ Error parsing sensor data: {e}")
        return None
    
    def calculate_tremor(self, acc_x, acc_y, acc_z):
        """Calculate tremor from accelerometer data"""
        # Calculate magnitude of acceleration
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        # Tremor is the variance of acceleration magnitude
        return np.var(acc_magnitude) if len(acc_magnitude) > 1 else 0.0
    
    def collect_baseline_data(self):
        """Collect baseline data from sensors"""
        print(f"\n🚀 Starting baseline data collection...")
        print(f"   ⏱️  Duration: {self.baseline_duration} seconds")
        print(f"   📊 Target samples: {self.total_samples}")
        print(f"   🧘 Please stay calm and relaxed during collection")
        print(f"   📱 Press Ctrl+C to stop early if needed")
        
        # Start data collection
        self.collecting = True
        data_thread = threading.Thread(target=self.read_sensor_data)
        data_thread.daemon = True
        data_thread.start()
        
        start_time = time.time()
        sample_count = 0
        
        try:
            while self.collecting and (time.time() - start_time) < self.baseline_duration:
                # Process data from queue
                while not self.data_queue.empty():
                    data = self.data_queue.get()
                    
                    # Calculate tremor from accelerometer data
                    tremor = self.calculate_tremor(
                        data['acc_x'], data['acc_y'], data['acc_z']
                    )
                    
                    # Store sensor data
                    self.sensor_data['timestamp'].append(data['timestamp'])
                    self.sensor_data['heart_rate'].append(data['heart_rate'])
                    self.sensor_data['eda'].append(data['eda'])
                    self.sensor_data['respiration'].append(data['respiration'])
                    self.sensor_data['temperature'].append(data['temperature'])
                    self.sensor_data['tremor'].append(tremor)
                    
                    sample_count += 1
                    
                    # Progress update every 10 seconds
                    elapsed = time.time() - start_time
                    if sample_count % 100 == 0:  # Every 10 seconds at 10 Hz
                        progress = (elapsed / self.baseline_duration) * 100
                        print(f"   📈 Progress: {elapsed:.1f}s / {self.baseline_duration}s ({progress:.1f}%) - Samples: {sample_count}")
                
                time.sleep(0.1)
            
            # Stop collection
            self.collecting = False
            data_thread.join(timeout=1)
            
            print(f"\n✅ Data collection completed!")
            print(f"   📊 Total samples collected: {sample_count}")
            print(f"   ⏱️  Duration: {time.time() - start_time:.1f} seconds")
            
            return sample_count > 0
            
        except KeyboardInterrupt:
            print(f"\n⏹️ Data collection stopped by user")
            self.collecting = False
            data_thread.join(timeout=1)
            return sample_count > 0
    
    def calculate_baseline(self):
        """Calculate baseline from collected data"""
        print(f"\n📊 Calculating baseline from collected data...")
        
        if not self.sensor_data['heart_rate']:
            print("❌ No data collected for baseline calculation")
            return False
        
        # Convert to numpy arrays
        hr_data = np.array(self.sensor_data['heart_rate'])
        eda_data = np.array(self.sensor_data['eda'])
        resp_data = np.array(self.sensor_data['respiration'])
        temp_data = np.array(self.sensor_data['temperature'])
        tremor_data = np.array(self.sensor_data['tremor'])
        
        # Remove invalid data (NaN, inf, extreme values)
        valid_mask = (
            np.isfinite(hr_data) & (hr_data > 30) & (hr_data < 200) &
            np.isfinite(eda_data) & (eda_data > 0) & (eda_data < 50) &
            np.isfinite(resp_data) & (resp_data > 5) & (resp_data < 40) &
            np.isfinite(temp_data) & (temp_data > 30) & (temp_data < 45) &
            np.isfinite(tremor_data) & (tremor_data >= 0)
        )
        
        if np.sum(valid_mask) < 50:  # Need at least 50 valid samples
            print("❌ Insufficient valid data for baseline calculation")
            return False
        
        # Calculate baseline statistics
        baseline = {
            'user_id': self.user_id,
            'timestamp': datetime.now().isoformat(),
            'heart_rate': {
                'mean': np.mean(hr_data[valid_mask]),
                'std': np.std(hr_data[valid_mask]),
                'median': np.median(hr_data[valid_mask])
            },
            'eda': {
                'mean': np.mean(eda_data[valid_mask]),
                'std': np.std(eda_data[valid_mask]),
                'median': np.median(eda_data[valid_mask])
            },
            'respiration': {
                'mean': np.mean(resp_data[valid_mask]),
                'std': np.std(resp_data[valid_mask]),
                'median': np.median(resp_data[valid_mask])
            },
            'temperature': {
                'mean': np.mean(temp_data[valid_mask]),
                'std': np.std(temp_data[valid_mask]),
                'median': np.median(temp_data[valid_mask])
            },
            'tremor': {
                'mean': np.mean(tremor_data[valid_mask]),
                'std': np.std(tremor_data[valid_mask]),
                'median': np.median(tremor_data[valid_mask])
            },
            'data_quality': {
                'total_samples': len(hr_data),
                'valid_samples': np.sum(valid_mask),
                'valid_percentage': (np.sum(valid_mask) / len(hr_data)) * 100
            }
        }
        
        self.baseline = baseline
        
        print(f"✅ Baseline calculated successfully!")
        print(f"   ❤️  Heart Rate: {baseline['heart_rate']['mean']:.1f} ± {baseline['heart_rate']['std']:.1f} BPM")
        print(f"   💧 EDA: {baseline['eda']['mean']:.1f} ± {baseline['eda']['std']:.1f} μS")
        print(f"   🫁 Respiration: {baseline['respiration']['mean']:.1f} ± {baseline['respiration']['std']:.1f} BPM")
        print(f"   🌡️  Temperature: {baseline['temperature']['mean']:.1f} ± {baseline['temperature']['std']:.1f}°C")
        print(f"   🤲 Tremor: {baseline['tremor']['mean']:.3f} ± {baseline['tremor']['std']:.3f}")
        print(f"   📊 Data Quality: {baseline['data_quality']['valid_percentage']:.1f}% valid samples")
        
        return True
    
    def save_baseline(self, filepath=None):
        """Save baseline to file"""
        if not self.baseline:
            print("❌ No baseline to save")
            return False
        
        if filepath is None:
            filepath = os.path.join(os.getcwd(), 'medical_baselines.pkl')
        
        try:
            # Load existing baselines if file exists
            existing_baselines = {}
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    existing_baselines = pickle.load(f)
            
            # Add new baseline
            existing_baselines[self.user_id] = self.baseline
            
            # Save updated baselines
            with open(filepath, 'wb') as f:
                pickle.dump(existing_baselines, f)
            
            print(f"✅ Baseline saved successfully!")
            print(f"   📁 File: {filepath}")
            print(f"   👤 User: {self.user_id}")
            print(f"   📊 Total users in file: {len(existing_baselines)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save baseline: {e}")
            return False
    
    def run_baseline_training(self):
        """Run complete baseline training process"""
        print(f"🎯 Starting Real-Time Baseline Training")
        print(f"=" * 50)
        
        # Connect to Arduino
        if not self.connect_arduino():
            return False
        
        try:
            # Collect baseline data
            if not self.collect_baseline_data():
                return False
            
            # Calculate baseline
            if not self.calculate_baseline():
                return False
            
            # Save baseline
            if not self.save_baseline():
                return False
            
            print(f"\n🎉 Baseline training completed successfully!")
            print(f"   👤 User ID: {self.user_id}")
            print(f"   📊 Baseline ready for panic attack detection")
            print(f"   🚀 You can now use the real-time predictor!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during baseline training: {e}")
            return False
        
        finally:
            self.disconnect_arduino()

def main():
    """Main function to run baseline training"""
    print("🏥 Real-Time Baseline Trainer for Panic Attack Detection")
    print("=" * 60)
    
    # Get user input
    user_id = input("👤 Enter your user ID (or press Enter for auto-generated): ").strip()
    if not user_id:
        user_id = None
    
    port = input("🔌 Enter Arduino port (default COM3): ").strip()
    if not port:
        port = 'COM3'
    
    # Create trainer
    trainer = RealTimeBaselineTrainer(port=port, user_id=user_id)
    
    # Run training
    success = trainer.run_baseline_training()
    
    if success:
        print(f"\n✅ Baseline training completed successfully!")
        print(f"   📁 Baseline saved to: medical_baselines.pkl")
        print(f"   🚀 Ready for real-time panic attack detection!")
    else:
        print(f"\n❌ Baseline training failed!")
        print(f"   💡 Check Arduino connection and try again")

if __name__ == "__main__":
    main()
