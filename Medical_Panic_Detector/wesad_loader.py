import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class WESADLoader:
    def __init__(self):
        # WESAD label mapping
        self.label_map = {
            0: 'Baseline',      # Baseline
            1: 'Stress',        # Stress
            2: 'Amusement',     # Amusement
            3: 'Meditation',    # Meditation
            4: 'Exercise',      # Exercise
            5: 'Other',         # Other states
            6: 'Panic',         # Panic Attack (our target)
            7: 'Other'          # Other states
        }
        
        # Sensor sampling rates
        self.sampling_rates = {
            'chest': 700,  # Chest sensor
            'wrist': 700,  # Wrist sensor
            'bvp': 64,     # Blood Volume Pulse
            'eda': 4,      # Electrodermal Activity
            'temp': 4,     # Temperature
            'acc': 32,     # Accelerometer
            'resp': 700    # Respiration
        }
    
    def load_wesad_subject(self, subject_id):
        """Load a single WESAD subject with proper signal extraction"""
        try:
            subject_file = f"WESAD/S{subject_id}/S{subject_id}.pkl"
            print(f"📁 Loading Subject S{subject_id}...")
            
            with open(subject_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # Extract signals and labels
            signals = data['signal']
            labels = data['label']
            subject = data['subject']
            
            print(f"  ✅ Loaded: {len(labels)} samples")
            print(f"  📊 Labels: {np.unique(labels)}")
            print(f"  🏷️  Subject: {subject}")
            
            # Extract individual sensor data from nested structure
            sensor_data = {}
            
            # Chest sensor data
            if 'chest' in signals:
                chest_data = signals['chest']
                if isinstance(chest_data, dict):
                    for sensor_name, sensor_values in chest_data.items():
                        if isinstance(sensor_values, np.ndarray):
                            sensor_data[f'chest_{sensor_name}'] = sensor_values
                            print(f"  📡 chest_{sensor_name}: shape {sensor_values.shape}")
            
            # Wrist sensor data
            if 'wrist' in signals:
                wrist_data = signals['wrist']
                if isinstance(wrist_data, dict):
                    for sensor_name, sensor_values in wrist_data.items():
                        if isinstance(sensor_values, np.ndarray):
                            sensor_data[f'wrist_{sensor_name}'] = sensor_values
                            print(f"  📡 wrist_{sensor_name}: shape {sensor_values.shape}")
            
            return {
                'subject_id': subject_id,
                'signals': sensor_data,
                'labels': labels,
                'subject_info': subject
            }
            
        except FileNotFoundError:
            print(f"  ❌ Subject S{subject_id} not found")
            return None
        except Exception as e:
            print(f"  ❌ Error loading S{subject_id}: {e}")
            return None
    
    def load_all_subjects(self):
        """Load all WESAD subjects"""
        subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
        all_data = []
        
        print("🏥 Loading Complete WESAD Dataset...")
        print("=" * 60)
        
        for subject_id in subject_ids:
            data = self.load_wesad_subject(subject_id)
            if data is not None:
                all_data.append(data)
        
        print(f"\n📊 Dataset Summary:")
        print(f"  ✅ Loaded {len(all_data)} subjects")
        print(f"  📈 Total samples: {sum(len(d['labels']) for d in all_data):,}")
        
        return all_data
    
    def extract_medical_features(self, signals, window_size=30, overlap=0.5):
        """Extract medical-grade features from physiological signals"""
        features = []
        
        # Get available signals
        available_signals = {k: v for k, v in signals.items() if len(v) > 0}
        
        if not available_signals:
            print("  ❌ No valid signals found!")
            return np.array([])
        
        # Get the minimum length across all signals
        min_length = min(len(signal) for signal in available_signals.values())
        print(f"  📏 Processing {min_length:,} samples")
        
        # Calculate window parameters (using 64Hz as base rate)
        window_samples = int(window_size * 64)
        step_size = int(window_samples * (1 - overlap))
        
        print(f"  🪟 Window: {window_samples} samples ({window_size}s), Step: {step_size} samples")
        
        for start in range(0, min_length - window_samples, step_size):
            end = start + window_samples
            
            window_features = []
            
            # Extract features from each sensor
            for sensor_name, signal in available_signals.items():
                window_signal = signal[start:end]
                
                # Clean signal (remove NaN and infinite values)
                window_signal = np.nan_to_num(window_signal, nan=0.0, posinf=0.0, neginf=0.0)
                
                if len(window_signal) == 0:
                    continue
                
                # Statistical features
                window_features.extend([
                    np.mean(window_signal),
                    np.std(window_signal),
                    np.var(window_signal),
                    np.min(window_signal),
                    np.max(window_signal),
                    np.median(window_signal),
                    np.percentile(window_signal, 25),
                    np.percentile(window_signal, 75),
                    np.percentile(window_signal, 90),
                    np.percentile(window_signal, 95)
                ])
                
                # Frequency domain features
                try:
                    fft = np.fft.fft(window_signal)
                    power_spectrum = np.abs(fft) ** 2
                    window_features.extend([
                        np.mean(power_spectrum),
                        np.std(power_spectrum),
                        np.sum(power_spectrum[:len(power_spectrum)//4]),  # Low freq power
                        np.sum(power_spectrum[len(power_spectrum)//4:len(power_spectrum)//2]),  # Mid freq power
                        np.sum(power_spectrum[len(power_spectrum)//2:])  # High freq power
                    ])
                except:
                    # If FFT fails, add zeros
                    window_features.extend([0.0] * 5)
            
            features.append(window_features)
        
        print(f"  ✅ Extracted {len(features)} feature windows")
        print(f"  📊 Features per window: {len(features[0]) if features else 0}")
        
        return np.array(features)
    
    def create_labels_for_windows(self, labels, window_size=30, overlap=0.5):
        """Create labels for each feature window"""
        window_samples = int(window_size * 64)
        step_size = int(window_samples * (1 - overlap))
        
        window_labels = []
        for start in range(0, len(labels) - window_samples, step_size):
            end = start + window_samples
            window_label = labels[start:end]
            
            # Use majority vote for window label
            unique, counts = np.unique(window_label, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            window_labels.append(majority_label)
        
        return np.array(window_labels)
    
    def prepare_medical_dataset(self, all_data, window_size=30):
        """Prepare the complete medical dataset for training"""
        print("\n🏥 Preparing Medical-Grade Dataset...")
        print("=" * 50)
        
        all_features = []
        all_labels = []
        subject_ids = []
        
        for data in all_data:
            print(f"\n📊 Processing Subject {data['subject_id']}...")
            
            # Extract features
            features = self.extract_medical_features(data['signals'], window_size)
            if len(features) == 0:
                print(f"  ⚠️  No features extracted for Subject {data['subject_id']}")
                continue
            
            # Create labels for windows
            labels = self.create_labels_for_windows(data['labels'], window_size)
            
            # Ensure features and labels have same length
            min_length = min(len(features), len(labels))
            features = features[:min_length]
            labels = labels[:min_length]
            
            all_features.append(features)
            all_labels.append(labels)
            subject_ids.extend([data['subject_id']] * min_length)
            
            print(f"  ✅ Subject {data['subject_id']}: {len(features)} windows")
        
        # Combine all data
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        subject_ids = np.array(subject_ids)
        
        print(f"\n📊 Final Dataset:")
        print(f"  📈 Total windows: {len(X)}")
        print(f"  🔢 Features per window: {X.shape[1]}")
        print(f"  🏷️  Unique labels: {np.unique(y)}")
        print(f"  📊 Label distribution: {np.bincount(y)}")
        
        return X, y, subject_ids

def main():
    """Test the WESAD loader"""
    print("🔬 Testing WESAD Data Loader")
    print("=" * 50)
    
    loader = WESADLoader()
    
    # Load all subjects
    all_data = loader.load_all_subjects()
    
    if not all_data:
        print("❌ No data loaded!")
        return
    
    # Prepare dataset
    X, y, subject_ids = loader.prepare_medical_dataset(all_data, window_size=30)
    
    print(f"\n✅ Dataset Preparation Complete!")
    print(f"  📊 Features shape: {X.shape}")
    print(f"  🏷️  Labels shape: {y.shape}")
    print(f"  👥 Subjects: {len(np.unique(subject_ids))}")

if __name__ == "__main__":
    main()
