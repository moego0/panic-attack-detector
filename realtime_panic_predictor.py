"""
Real-Time Panic Attack Predictor
================================

This script reads real-time sensor data from Arduino and uses the trained
models to predict panic attacks. It combines the universal models with
the user's personal baseline for accurate detection.

Author: Medical Panic Detection System
Date: 2025
"""

import serial
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

class RealTimePanicPredictor:
    """Real-time panic attack predictor using trained models"""
    
    def __init__(self, port='COM3', baudrate=9600, user_id=None, models_path=None):
        """
        Initialize the real-time panic predictor
        
        Args:
            port (str): Serial port for Arduino connection
            baudrate (int): Baud rate for serial communication
            user_id (str): Unique identifier for the user
            models_path (str): Path to the models directory
        """
        self.port = port
        self.baudrate = baudrate
        self.user_id = user_id
        self.models_path = models_path or r"E:\panic attack detector\models"
        
        # Data collection parameters
        self.window_size = 30  # 30 seconds window for prediction
        self.sampling_rate = 10  # 10 Hz sampling rate
        self.window_samples = self.window_size * self.sampling_rate
        
        # Data storage
        self.sensor_data = {
            'timestamp': [],
            'heart_rate': [],
            'eda': [],
            'respiration': [],
            'temperature': [],
            'tremor': []
        }
        
        # Models and baselines
        self.models = {}
        self.baseline = None
        self.scaler = None
        self.feature_selector = None
        self.thresholds = None
        
        # Serial connection
        self.serial_conn = None
        self.data_queue = queue.Queue()
        self.monitoring = False
        
        # Prediction history
        self.prediction_history = []
        self.alert_history = []
        
        print(f"🎯 Real-Time Panic Predictor Initialized")
        print(f"   👤 User ID: {self.user_id}")
        print(f"   📊 Window Size: {self.window_size} seconds")
        print(f"   📈 Sampling Rate: {self.sampling_rate} Hz")
        print(f"   🤖 Models Path: {self.models_path}")
    
    def load_models(self):
        """Load all trained models and baselines"""
        print(f"📚 Loading trained models and baselines...")
        
        try:
            # Load baselines
            baselines_path = os.path.join(self.models_path, 'medical_baselines.pkl')
            if os.path.exists(baselines_path):
                with open(baselines_path, 'rb') as f:
                    all_baselines = pickle.load(f)
                
                if self.user_id in all_baselines:
                    self.baseline = all_baselines[self.user_id]
                    print(f"✅ Loaded baseline for user: {self.user_id}")
                else:
                    print(f"❌ No baseline found for user: {self.user_id}")
                    print(f"💡 Available users: {list(all_baselines.keys())}")
                    return False
            else:
                print(f"❌ Baselines file not found: {baselines_path}")
                return False
            
            # Load scaler
            scaler_path = os.path.join(self.models_path, 'medical_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Loaded scaler")
            else:
                print(f"❌ Scaler file not found: {scaler_path}")
                return False
            
            # Load feature selector
            selector_path = os.path.join(self.models_path, 'medical_feature_selector.pkl')
            if os.path.exists(selector_path):
                with open(selector_path, 'rb') as f:
                    self.feature_selector = pickle.load(f)
                print(f"✅ Loaded feature selector")
            else:
                print(f"❌ Feature selector file not found: {selector_path}")
                return False
            
            # Load thresholds
            thresholds_path = os.path.join(self.models_path, 'medical_thresholds.pkl')
            if os.path.exists(thresholds_path):
                with open(thresholds_path, 'rb') as f:
                    self.thresholds = pickle.load(f)
                print(f"✅ Loaded clinical thresholds")
            else:
                print(f"❌ Thresholds file not found: {thresholds_path}")
                return False
            
            # Load individual models
            model_files = {
                'ensemble': 'medical_ensemble_model.pkl',
                'random_forest': 'medical_random_forest_model.pkl',
                'gradient_boosting': 'medical_gradient_boosting_model.pkl',
                'neural_network': 'medical_neural_network_model.pkl',
                'svm': 'medical_svm_model.pkl',
                'logistic_regression': 'medical_logistic_regression_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = os.path.join(self.models_path, filename)
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"✅ Loaded {model_name} model")
                else:
                    print(f"❌ {model_name} model not found: {model_path}")
                    return False
            
            print(f"✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def connect_arduino(self):
        """Connect to Arduino via serial port"""
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
        """Disconnect from Arduino"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("🔌 Arduino disconnected")
    
    def read_sensor_data(self):
        """Read sensor data from Arduino in a separate thread"""
        while self.monitoring:
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
        """Parse sensor data from Arduino"""
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
    
    def extract_features(self, window_data):
        """Extract features from sensor window data"""
        try:
            # Get baseline values
            hr_baseline = self.baseline['heart_rate']['mean']
            eda_baseline = self.baseline['eda']['mean']
            resp_baseline = self.baseline['respiration']['mean']
            temp_baseline = self.baseline['temperature']['mean']
            tremor_baseline = self.baseline['tremor']['mean']
            
            # Extract basic features
            features = []
            
            # Heart rate features
            hr_data = np.array(window_data['heart_rate'])
            features.extend([
                np.mean(hr_data),
                np.std(hr_data),
                np.max(hr_data),
                np.min(hr_data),
                np.median(hr_data),
                np.percentile(hr_data, 25),
                np.percentile(hr_data, 75),
                np.mean(hr_data) - hr_baseline,  # Deviation from baseline
                np.std(hr_data) / hr_baseline if hr_baseline > 0 else 0,  # Relative variability
            ])
            
            # EDA features
            eda_data = np.array(window_data['eda'])
            features.extend([
                np.mean(eda_data),
                np.std(eda_data),
                np.max(eda_data),
                np.min(eda_data),
                np.median(eda_data),
                np.percentile(eda_data, 25),
                np.percentile(eda_data, 75),
                np.mean(eda_data) - eda_baseline,  # Deviation from baseline
                np.std(eda_data) / eda_baseline if eda_baseline > 0 else 0,  # Relative variability
            ])
            
            # Respiration features
            resp_data = np.array(window_data['respiration'])
            features.extend([
                np.mean(resp_data),
                np.std(resp_data),
                np.max(resp_data),
                np.min(resp_data),
                np.median(resp_data),
                np.percentile(resp_data, 25),
                np.percentile(resp_data, 75),
                np.mean(resp_data) - resp_baseline,  # Deviation from baseline
                np.std(resp_data) / resp_baseline if resp_baseline > 0 else 0,  # Relative variability
            ])
            
            # Temperature features
            temp_data = np.array(window_data['temperature'])
            features.extend([
                np.mean(temp_data),
                np.std(temp_data),
                np.max(temp_data),
                np.min(temp_data),
                np.median(temp_data),
                np.percentile(temp_data, 25),
                np.percentile(temp_data, 75),
                np.mean(temp_data) - temp_baseline,  # Deviation from baseline
                np.std(temp_data) / temp_baseline if temp_baseline > 0 else 0,  # Relative variability
            ])
            
            # Tremor features
            tremor_data = np.array(window_data['tremor'])
            features.extend([
                np.mean(tremor_data),
                np.std(tremor_data),
                np.max(tremor_data),
                np.min(tremor_data),
                np.median(tremor_data),
                np.percentile(tremor_data, 25),
                np.percentile(tremor_data, 75),
                np.mean(tremor_data) - tremor_baseline,  # Deviation from baseline
                np.std(tremor_data) / tremor_baseline if tremor_baseline > 0 else 0,  # Relative variability
            ])
            
            # Cross-signal features
            features.extend([
                np.corrcoef(hr_data, eda_data)[0, 1] if len(hr_data) > 1 else 0,
                np.corrcoef(hr_data, resp_data)[0, 1] if len(hr_data) > 1 else 0,
                np.corrcoef(eda_data, resp_data)[0, 1] if len(eda_data) > 1 else 0,
                np.corrcoef(hr_data, temp_data)[0, 1] if len(hr_data) > 1 else 0,
                np.corrcoef(eda_data, temp_data)[0, 1] if len(eda_data) > 1 else 0,
            ])
            
            # Convert to numpy array and handle NaN/inf
            features = np.array(features)
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return features
            
        except Exception as e:
            print(f"⚠️ Error extracting features: {e}")
            return None
    
    def predict_panic_attack(self, features):
        """Predict panic attack using trained models"""
        try:
            if features is None:
                return None, 0.0
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Select features
            features_selected = self.feature_selector.transform(features_scaled)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(features_selected)[0]
                        predictions[model_name] = model.predict(features_selected)[0]
                        probabilities[model_name] = prob[1] if len(prob) > 1 else prob[0]
                    else:
                        pred = model.predict(features_selected)[0]
                        predictions[model_name] = pred
                        probabilities[model_name] = float(pred)
                except Exception as e:
                    print(f"⚠️ Error with {model_name} model: {e}")
                    predictions[model_name] = 0
                    probabilities[model_name] = 0.0
            
            # Calculate ensemble prediction
            ensemble_prob = np.mean(list(probabilities.values()))
            ensemble_pred = 1 if ensemble_prob > 0.5 else 0
            
            return {
                'prediction': ensemble_pred,
                'probability': ensemble_prob,
                'individual_predictions': predictions,
                'individual_probabilities': probabilities
            }
            
        except Exception as e:
            print(f"⚠️ Error in prediction: {e}")
            return None
    
    def check_clinical_thresholds(self, window_data):
        """Check clinical thresholds for panic attack symptoms"""
        try:
            # Get current values
            hr_current = np.mean(window_data['heart_rate'])
            eda_current = np.mean(window_data['eda'])
            resp_current = np.mean(window_data['respiration'])
            temp_current = np.mean(window_data['temperature'])
            tremor_current = np.mean(window_data['tremor'])
            
            # Get baseline values
            hr_baseline = self.baseline['heart_rate']['mean']
            eda_baseline = self.baseline['eda']['mean']
            resp_baseline = self.baseline['respiration']['mean']
            temp_baseline = self.baseline['temperature']['mean']
            tremor_baseline = self.baseline['tremor']['mean']
            
            # Check thresholds
            symptoms = []
            
            # Heart rate threshold (25% above baseline)
            if hr_current > hr_baseline * 1.25:
                symptoms.append(f"HR: {hr_current:.1f} > {hr_baseline * 1.25:.1f} BPM")
            
            # EDA threshold (30% above baseline)
            if eda_current > eda_baseline * 1.30:
                symptoms.append(f"EDA: {eda_current:.1f} > {eda_baseline * 1.30:.1f} μS")
            
            # Respiration threshold (20% above baseline)
            if resp_current > resp_baseline * 1.20:
                symptoms.append(f"Resp: {resp_current:.1f} > {resp_baseline * 1.20:.1f} BPM")
            
            # Temperature threshold (5% below baseline)
            if temp_current < temp_baseline * 0.95:
                symptoms.append(f"Temp: {temp_current:.1f} < {temp_baseline * 0.95:.1f}°C")
            
            # Tremor threshold (50% above baseline)
            if tremor_current > tremor_baseline * 1.50:
                symptoms.append(f"Tremor: {tremor_current:.3f} > {tremor_baseline * 1.50:.3f}")
            
            return {
                'symptoms': symptoms,
                'symptom_count': len(symptoms),
                'panic_probability': len(symptoms) / 5.0
            }
            
        except Exception as e:
            print(f"⚠️ Error checking clinical thresholds: {e}")
            return {'symptoms': [], 'symptom_count': 0, 'panic_probability': 0.0}
    
    def process_prediction(self, prediction_result, clinical_result):
        """Process prediction results and generate alerts"""
        try:
            # Combine ML prediction and clinical assessment
            ml_prob = prediction_result['probability'] if prediction_result else 0.0
            clinical_prob = clinical_result['panic_probability']
            
            # Weighted combination (70% ML, 30% clinical)
            combined_prob = 0.7 * ml_prob + 0.3 * clinical_prob
            
            # Determine alert level
            if combined_prob >= 0.8:
                alert_level = "CRITICAL"
                message = "🚨 PANIC ATTACK DETECTED! Seek help immediately!"
            elif combined_prob >= 0.6:
                alert_level = "HIGH"
                message = "⚠️ HIGH STRESS! Consider relaxation techniques"
            elif combined_prob >= 0.4:
                alert_level = "MEDIUM"
                message = "🔔 ELEVATED STRESS! Monitor your condition"
            else:
                alert_level = "NORMAL"
                message = "✅ Normal stress levels - You're doing well!"
            
            # Create prediction record
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'ml_probability': ml_prob,
                'clinical_probability': clinical_prob,
                'combined_probability': combined_prob,
                'alert_level': alert_level,
                'message': message,
                'symptoms': clinical_result['symptoms'],
                'symptom_count': clinical_result['symptom_count']
            }
            
            # Store prediction
            self.prediction_history.append(prediction_record)
            
            # Keep only last 100 predictions
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            return prediction_record
            
        except Exception as e:
            print(f"⚠️ Error processing prediction: {e}")
            return None
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        print(f"\n🚀 Starting real-time panic attack monitoring...")
        print(f"   ⏱️  Window size: {self.window_size} seconds")
        print(f"   📊 Sampling rate: {self.sampling_rate} Hz")
        print(f"   🧘 Stay calm and relaxed during monitoring")
        print(f"   📱 Press Ctrl+C to stop monitoring")
        
        # Start data collection
        self.monitoring = True
        data_thread = threading.Thread(target=self.read_sensor_data)
        data_thread.daemon = True
        data_thread.start()
        
        try:
            while self.monitoring:
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
                    
                    # Keep only last window_size seconds of data
                    current_time = data['timestamp']
                    cutoff_time = current_time - self.window_size
                    
                    # Remove old data
                    valid_indices = [i for i, t in enumerate(self.sensor_data['timestamp']) if t >= cutoff_time]
                    
                    for key in self.sensor_data:
                        self.sensor_data[key] = [self.sensor_data[key][i] for i in valid_indices]
                    
                    # Check if we have enough data for prediction
                    if len(self.sensor_data['heart_rate']) >= self.window_samples:
                        # Extract features
                        features = self.extract_features(self.sensor_data)
                        
                        # Get ML prediction
                        ml_prediction = self.predict_panic_attack(features)
                        
                        # Check clinical thresholds
                        clinical_result = self.check_clinical_thresholds(self.sensor_data)
                        
                        # Process prediction
                        prediction_record = self.process_prediction(ml_prediction, clinical_result)
                        
                        if prediction_record:
                            # Display results
                            print(f"\n📊 {datetime.now().strftime('%H:%M:%S')} - {prediction_record['message']}")
                            print(f"   🤖 ML Probability: {prediction_record['ml_probability']:.1%}")
                            print(f"   🏥 Clinical Probability: {prediction_record['clinical_probability']:.1%}")
                            print(f"   🎯 Combined Probability: {prediction_record['combined_probability']:.1%}")
                            print(f"   🚨 Alert Level: {prediction_record['alert_level']}")
                            
                            if prediction_record['symptoms']:
                                print(f"   ⚠️  Symptoms: {', '.join(prediction_record['symptoms'])}")
                            
                            # Store alert if critical or high
                            if prediction_record['alert_level'] in ['CRITICAL', 'HIGH']:
                                self.alert_history.append(prediction_record)
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Monitoring stopped by user")
            self.monitoring = False
            data_thread.join(timeout=1)
    
    def run_monitoring(self):
        """Run complete monitoring process"""
        print(f"🎯 Starting Real-Time Panic Attack Monitoring")
        print(f"=" * 50)
        
        # Load models
        if not self.load_models():
            return False
        
        # Connect to Arduino
        if not self.connect_arduino():
            return False
        
        try:
            # Start monitoring
            self.start_monitoring()
            
            print(f"\n✅ Monitoring completed!")
            print(f"   📊 Total predictions: {len(self.prediction_history)}")
            print(f"   🚨 Total alerts: {len(self.alert_history)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
            return False
        
        finally:
            self.disconnect_arduino()

def main():
    """Main function to run panic attack monitoring"""
    print("🏥 Real-Time Panic Attack Predictor")
    print("=" * 40)
    
    # Get user input
    user_id = input("👤 Enter your user ID: ").strip()
    if not user_id:
        print("❌ User ID is required!")
        return
    
    port = input("🔌 Enter Arduino port (default COM3): ").strip()
    if not port:
        port = 'COM3'
    
    models_path = input("📁 Enter models path (default: E:\\panic attack detector\\models): ").strip()
    if not models_path:
        models_path = r"E:\panic attack detector\models"
    
    # Create predictor
    predictor = RealTimePanicPredictor(
        port=port, 
        user_id=user_id, 
        models_path=models_path
    )
    
    # Run monitoring
    success = predictor.run_monitoring()
    
    if success:
        print(f"\n✅ Monitoring completed successfully!")
    else:
        print(f"\n❌ Monitoring failed!")
        print(f"   💡 Check Arduino connection and models path")

if __name__ == "__main__":
    main()
