import os
import sys
import numpy as np
import pandas as pd
import pickle
import joblib
import time
import threading
import queue
import json
from datetime import datetime
import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

class MedicalRealtimeDetector:
    """
    Medical-grade real-time panic attack detection system.
    Integrates with Arduino via Bluetooth for continuous monitoring.
    """
    
    def __init__(self, port='COM3', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.data_queue = queue.Queue()
        self.is_running = False
        
        # Load trained models
        self.load_medical_models()
        
        # Data buffers for real-time processing
        self.buffer_size = 7000  # 10 seconds at 700Hz
        self.buffer_overlap = 3500  # 50% overlap
        self.signal_buffers = {
            'ecg': [],
            'eda': [],
            'resp': [],
            'emg': [],
            'temp': [],
            'bvp': [],
            'wrist_eda': [],
            'wrist_temp': [],
            'acc': []
        }
        
        # Detection results
        self.detection_history = []
        self.current_baseline = None
        self.panic_alert_count = 0
        
        # Medical thresholds
        self.alert_threshold = 0.7  # Probability threshold for panic alert
        self.confirmation_window = 3  # Seconds to confirm panic attack
        
    def load_medical_models(self):
        """Load pre-trained medical models"""
        try:
            print("🏥 Loading Medical-Grade Models...")
            
            # Load ensemble model
            self.ensemble_model = joblib.load(r'E:\panic attack detector\models\medical_ensemble_model.pkl')
            
            # Load preprocessors
            self.scaler = joblib.load(r'E:\panic attack detector\models\medical_scaler.pkl')
            self.feature_selector = joblib.load(r'E:\panic attack detector\models\medical_feature_selector.pkl')
            
            # Load baselines and thresholds
            with open(r'E:\panic attack detector\models\medical_baselines.pkl', 'rb') as f:
                self.baseline_data = pickle.load(f)
            
            with open(r'E:\panic attack detector\models\medical_thresholds.pkl', 'rb') as f:
                self.thresholds = pickle.load(f)
            
            print("  ✅ All medical models loaded successfully!")
            
        except Exception as e:
            print(f"  ❌ Error loading models: {e}")
            raise
    
    def connect_arduino(self):
        """Connect to Arduino via Bluetooth"""
        try:
            print(f"📡 Connecting to Arduino on {self.port}...")
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            time.sleep(2)  # Wait for connection to stabilize
            print("  ✅ Arduino connected successfully!")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to connect to Arduino: {e}")
            return False
    
    def disconnect_arduino(self):
        """Disconnect from Arduino"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("📡 Arduino disconnected.")
    
    def read_arduino_data(self):
        """Read data from Arduino in separate thread"""
        while self.is_running:
            try:
                if self.serial_connection and self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8').strip()
                    if line:
                        self.data_queue.put(line)
                time.sleep(0.001)  # 1ms delay
            except Exception as e:
                print(f"Error reading Arduino data: {e}")
                break
    
    def parse_arduino_data(self, data_line):
        """Parse data from Arduino format"""
        try:
            # Expected format: "ECG,EDA,RESP,EMG,TEMP,BVP,WEDA,WTEMP,ACCX,ACCY,ACCZ"
            values = data_line.split(',')
            if len(values) >= 11:
                return {
                    'ecg': float(values[0]),
                    'eda': float(values[1]),
                    'resp': float(values[2]),
                    'emg': float(values[3]),
                    'temp': float(values[4]),
                    'bvp': float(values[5]),
                    'wrist_eda': float(values[6]),
                    'wrist_temp': float(values[7]),
                    'acc_x': float(values[8]),
                    'acc_y': float(values[9]),
                    'acc_z': float(values[10]),
                    'timestamp': time.time()
                }
        except Exception as e:
            print(f"Error parsing data: {e}")
        return None
    
    def add_to_buffer(self, data):
        """Add new data to signal buffers"""
        for signal_name in self.signal_buffers:
            if signal_name in data:
                self.signal_buffers[signal_name].append(data[signal_name])
                
                # Keep buffer size manageable
                if len(self.signal_buffers[signal_name]) > self.buffer_size * 2:
                    self.signal_buffers[signal_name] = self.signal_buffers[signal_name][-self.buffer_size:]
    
    def extract_realtime_features(self, window_data, baseline):
        """Extract features for real-time detection"""
        features = []
        
        # 1. Clinical features (15 features)
        clinical_features = self._extract_clinical_features_realtime(window_data, baseline)
        features.extend(clinical_features)
        
        # 2. Statistical features (72 features)
        for signal_name, signal_data in window_data.items():
            if len(signal_data) > 0:
                stat_features = self._extract_statistical_features_realtime(signal_data)
                features.extend(stat_features)
            else:
                features.extend([0] * 8)
        
        # 3. Frequency features (45 features)
        for signal_name, signal_data in window_data.items():
            if len(signal_data) > 0:
                freq_features = self._extract_frequency_features_realtime(signal_data)
                features.extend(freq_features)
            else:
                features.extend([0] * 5)
        
        # 4. Time series features (54 features)
        for signal_name, signal_data in window_data.items():
            if len(signal_data) > 0:
                time_features = self._extract_time_series_features_realtime(signal_data)
                features.extend(time_features)
            else:
                features.extend([0] * 6)
        
        # 5. Cross-signal features (10 features)
        cross_features = self._extract_cross_signal_features_realtime(window_data)
        features.extend(cross_features)
        
        return np.array(features)
    
    def _extract_clinical_features_realtime(self, window_data, baseline):
        """Extract clinical features for real-time detection"""
        features = []
        
        # Heart rate features
        if 'chest_ecg' in window_data and len(window_data['chest_ecg']) > 0:
            current_hr = self._calculate_heart_rate_realtime(window_data['chest_ecg'])
            baseline_hr = baseline.get('heart_rate', 70)
            features.extend([
                current_hr,
                current_hr - baseline_hr,
                1 if current_hr >= baseline_hr + self.thresholds['heart_rate']['sudden_increase'] else 0,
                1 if current_hr >= self.thresholds['heart_rate']['absolute_threshold'] else 0
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # HRV features
        if 'chest_ecg' in window_data and len(window_data['chest_ecg']) > 0:
            current_hrv = self._calculate_hrv_realtime(window_data['chest_ecg'])
            baseline_hrv = baseline.get('hrv', 0.03)
            features.extend([
                current_hrv,
                1 if current_hrv <= baseline_hrv * (1 - self.thresholds['hrv']['drop_percentage']) else 0
            ])
        else:
            features.extend([0, 0])
        
        # EDA features
        if 'chest_eda' in window_data and len(window_data['chest_eda']) > 0:
            current_eda_rate = self._calculate_eda_rate_realtime(window_data['chest_eda'])
            baseline_eda_rate = baseline.get('eda', {}).get('rate_of_change', 0)
            features.extend([
                current_eda_rate,
                1 if current_eda_rate >= baseline_eda_rate + self.thresholds['eda']['spike_threshold'] else 0
            ])
        else:
            features.extend([0, 0])
        
        # Breathing rate features
        if 'chest_resp' in window_data and len(window_data['chest_resp']) > 0:
            current_br = self._calculate_breathing_rate_realtime(window_data['chest_resp'])
            features.extend([
                current_br,
                1 if current_br >= self.thresholds['breathing_rate']['threshold'] else 0
            ])
        else:
            features.extend([0, 0])
        
        # Tremor features
        if 'wrist_acc' in window_data and len(window_data['wrist_acc']) > 0:
            tremor_metrics = self._calculate_tremor_realtime(window_data['wrist_acc'])
            baseline_tremor = baseline.get('tremor', {'variance': 0})
            features.extend([
                tremor_metrics['variance'],
                1 if tremor_metrics['variance'] >= baseline_tremor['variance'] * self.thresholds['tremor']['variance_threshold'] else 0
            ])
        else:
            features.extend([0, 0])
        
        # Temperature features
        if 'wrist_temp' in window_data and len(window_data['wrist_temp']) > 0:
            current_temp = np.mean(window_data['wrist_temp'])
            baseline_temp = baseline.get('skin_temp', {}).get('mean', 36.5)
            features.extend([
                current_temp,
                1 if current_temp <= baseline_temp - self.thresholds['skin_temp']['drop_threshold'] else 0
            ])
        else:
            features.extend([0, 0])
        
        return features
    
    def _calculate_heart_rate_realtime(self, ecg_signal, fs=700):
        """Calculate heart rate for real-time detection"""
        try:
            from scipy import signal
            peaks, _ = signal.find_peaks(ecg_signal, height=np.std(ecg_signal), distance=fs//3)
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / fs
                return 60 / np.mean(rr_intervals)
            return 70
        except:
            return 70
    
    def _calculate_hrv_realtime(self, ecg_signal, fs=700):
        """Calculate HRV for real-time detection"""
        try:
            from scipy import signal
            peaks, _ = signal.find_peaks(ecg_signal, height=np.std(ecg_signal), distance=fs//3)
            if len(peaks) > 2:
                rr_intervals = np.diff(peaks) / fs
                return np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            return 0.03
        except:
            return 0.03
    
    def _calculate_eda_rate_realtime(self, eda_signal, window_size=10):
        """Calculate EDA rate for real-time detection"""
        try:
            if len(eda_signal) < window_size:
                return 0
            return (eda_signal[-1] - eda_signal[0]) / window_size
        except:
            return 0
    
    def _calculate_breathing_rate_realtime(self, resp_signal, fs=700):
        """Calculate breathing rate for real-time detection"""
        try:
            from scipy import signal
            peaks, _ = signal.find_peaks(resp_signal, height=np.std(resp_signal), distance=fs//2)
            if len(peaks) > 1:
                breathing_intervals = np.diff(peaks) / fs
                return 60 / np.mean(breathing_intervals)
            return 15
        except:
            return 15
    
    def _calculate_tremor_realtime(self, acc_signal):
        """Calculate tremor for real-time detection"""
        try:
            motion_variance = np.var(acc_signal, axis=0) if acc_signal.ndim > 1 else np.var(acc_signal)
            total_variance = np.sum(motion_variance)
            return {'variance': total_variance}
        except:
            return {'variance': 0}
    
    def _extract_statistical_features_realtime(self, signal):
        """Extract statistical features for real-time detection"""
        if len(signal) == 0:
            return [0] * 8
        
        try:
            signal_clean = signal[np.isfinite(signal)]
            if len(signal_clean) == 0:
                return [0] * 8
            
            features = [
                np.mean(signal_clean),
                np.std(signal_clean),
                np.var(signal_clean),
                np.min(signal_clean),
                np.max(signal_clean),
                np.ptp(signal_clean),
                np.mean(np.abs(signal_clean - np.mean(signal_clean))),  # MAD
                np.sum(signal_clean > np.mean(signal_clean)) / len(signal_clean)  # Above mean ratio
            ]
            
            return [0 if not np.isfinite(f) else f for f in features]
        except:
            return [0] * 8
    
    def _extract_frequency_features_realtime(self, signal):
        """Extract frequency features for real-time detection"""
        if len(signal) == 0:
            return [0] * 5
        
        try:
            signal_clean = signal[np.isfinite(signal)]
            if len(signal_clean) < 4:
                return [0] * 5
            
            from scipy import signal as sp_signal
            freqs, psd = sp_signal.welch(signal_clean, fs=700, nperseg=min(len(signal_clean)//4, 1024))
            
            peak_freq = freqs[np.argmax(psd)]
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0
            
            cumsum_psd = np.cumsum(psd)
            rolloff_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
            
            zero_crossing_rate = np.sum(np.diff(np.sign(signal_clean)) != 0) / len(signal_clean)
            
            features = [peak_freq, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate]
            return [0 if not np.isfinite(f) else f for f in features]
        except:
            return [0] * 5
    
    def _extract_time_series_features_realtime(self, signal):
        """Extract time series features for real-time detection"""
        if len(signal) == 0:
            return [0] * 6
        
        try:
            signal_clean = signal[np.isfinite(signal)]
            if len(signal_clean) == 0:
                return [0] * 6
            
            x = np.arange(len(signal_clean))
            slope = np.polyfit(x, signal_clean, 1)[0] if len(signal_clean) > 1 else 0
            
            autocorr = np.corrcoef(signal_clean[:-1], signal_clean[1:])[0, 1] if len(signal_clean) > 1 else 0
            energy = np.sum(signal_clean ** 2)
            
            hist, _ = np.histogram(signal_clean, bins=10)
            hist = hist / np.sum(hist)
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            
            zero_crossings = np.sum(np.diff(np.sign(signal_clean)) != 0)
            mad = np.mean(np.abs(signal_clean - np.mean(signal_clean)))
            
            features = [slope, autocorr, energy, entropy, zero_crossings, mad]
            return [0 if not np.isfinite(f) else f for f in features]
        except:
            return [0] * 6
    
    def _extract_cross_signal_features_realtime(self, window_data):
        """Extract cross-signal features for real-time detection"""
        features = []
        
        signals = {}
        for name, data in window_data.items():
            if len(data) > 0:
                signals[name] = data.flatten()
        
        signal_pairs = [
            ('chest_ecg', 'chest_resp'),
            ('chest_eda', 'wrist_eda'),
            ('chest_temp', 'wrist_temp'),
            ('wrist_bvp', 'chest_ecg'),
            ('wrist_acc', 'chest_emg'),
            ('chest_ecg', 'wrist_bvp'),
            ('chest_resp', 'wrist_acc'),
            ('chest_eda', 'wrist_temp'),
            ('chest_emg', 'wrist_eda'),
            ('chest_temp', 'wrist_acc')
        ]
        
        for sig1, sig2 in signal_pairs:
            if sig1 in signals and sig2 in signals:
                try:
                    min_len = min(len(signals[sig1]), len(signals[sig2]))
                    corr = np.corrcoef(signals[sig1][:min_len], signals[sig2][:min_len])[0, 1]
                    features.append(corr if not np.isnan(corr) else 0)
                except:
                    features.append(0)
            else:
                features.append(0)
        
        return features
    
    def detect_panic_realtime(self, baseline):
        """Perform real-time panic attack detection"""
        if not all(len(buffer) >= self.buffer_size for buffer in self.signal_buffers.values()):
            return None
        
        try:
            # Get window data
            window_data = {}
            for signal_name, buffer in self.signal_buffers.items():
                window_data[signal_name] = np.array(buffer[-self.buffer_size:])
            
            # Extract features
            features = self.extract_realtime_features(window_data, baseline)
            
            if len(features) == 0:
                return None
            
            # Clean features
            features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale and select features
            features_scaled = self.scaler.transform(features_clean.reshape(1, -1))
            features_selected = self.feature_selector.transform(features_scaled)
            
            # Get prediction
            prediction = self.ensemble_model.predict(features_selected)[0]
            probability = self.ensemble_model.predict_proba(features_selected)[0][1]
            
            # Determine confidence
            if probability > 0.8 or probability < 0.2:
                confidence = 'high'
            elif probability > 0.6 or probability < 0.4:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            result = {
                'timestamp': datetime.now(),
                'prediction': int(prediction),
                'probability': float(probability),
                'confidence': confidence,
                'interpretation': 'PANIC ATTACK DETECTED' if prediction == 1 else 'Normal State',
                'alert_level': 'CRITICAL' if probability > self.alert_threshold else 'NORMAL'
            }
            
            return result
            
        except Exception as e:
            print(f"Error in real-time detection: {e}")
            return None
    
    def start_monitoring(self, subject_id='default'):
        """Start real-time monitoring"""
        print("🚨 Starting Medical-Grade Real-Time Monitoring...")
        
        # Get baseline for subject
        baseline_key = f'subject_{subject_id}'
        if baseline_key in self.baseline_data:
            self.current_baseline = self.baseline_data[baseline_key]
        else:
            print(f"⚠️ No baseline found for subject {subject_id}, using default")
            self.current_baseline = {
                'heart_rate': 70,
                'hrv': 0.03,
                'eda': {'mean': 5.0, 'rate_of_change': 0.01},
                'breathing_rate': 15,
                'tremor': {'variance': 1.0},
                'skin_temp': {'mean': 36.5}
            }
        
        # Connect to Arduino
        if not self.connect_arduino():
            print("❌ Cannot start monitoring without Arduino connection")
            return
        
        # Start data reading thread
        self.is_running = True
        data_thread = threading.Thread(target=self.read_arduino_data)
        data_thread.daemon = True
        data_thread.start()
        
        print("  ✅ Real-time monitoring started!")
        print("  📊 Processing 10-second windows with 50% overlap")
        print("  🚨 Panic detection active...")
        
        try:
            while self.is_running:
                # Process data from queue
                while not self.data_queue.empty():
                    data_line = self.data_queue.get()
                    data = self.parse_arduino_data(data_line)
                    
                    if data:
                        self.add_to_buffer(data)
                
                # Perform detection if we have enough data
                if all(len(buffer) >= self.buffer_size for buffer in self.signal_buffers.values()):
                    result = self.detect_panic_realtime(self.current_baseline)
                    
                    if result:
                        self.detection_history.append(result)
                        
                        # Print result
                        timestamp = result['timestamp'].strftime("%H:%M:%S")
                        print(f"[{timestamp}] {result['interpretation']} "
                              f"(Prob: {result['probability']:.3f}, "
                              f"Conf: {result['confidence']})")
                        
                        # Check for panic alert
                        if result['alert_level'] == 'CRITICAL':
                            self.panic_alert_count += 1
                            print(f"🚨 CRITICAL ALERT #{self.panic_alert_count}: "
                                  f"Panic attack detected with {result['probability']:.1%} confidence!")
                            
                            # Save alert to file
                            self.save_panic_alert(result)
                
                time.sleep(0.1)  # 100ms processing interval
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitoring...")
            self.is_running = False
        
        finally:
            self.disconnect_arduino()
            print("✅ Monitoring stopped.")
    
    def save_panic_alert(self, result):
        """Save panic alert to file"""
        alert_data = {
            'timestamp': result['timestamp'].isoformat(),
            'prediction': result['prediction'],
            'probability': result['probability'],
            'confidence': result['confidence'],
            'alert_level': result['alert_level']
        }
        
        with open('realtime/panic_alerts.json', 'a') as f:
            f.write(json.dumps(alert_data) + '\n')
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False

def main():
    """Main function for real-time detection"""
    print("🏥 Medical-Grade Real-Time Panic Attack Detection")
    print("Arduino Bluetooth Integration")
    print("="*60)
    
    # Initialize detector
    detector = MedicalRealtimeDetector(port='COM3', baudrate=9600)
    
    # Start monitoring
    try:
        detector.start_monitoring(subject_id='default')
    except Exception as e:
        print(f"Error in monitoring: {e}")

if __name__ == "__main__":
    main()
