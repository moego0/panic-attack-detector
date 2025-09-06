import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

class PanicAttackDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.feature_names = []
        
    def load_wesad_data(self, subject_ids=None):
        """Load WESAD dataset for specified subjects"""
        if subject_ids is None:
            subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
        
        all_data = []
        all_labels = []
        
        for subject_id in subject_ids:
            try:
                subject_file = f"WESAD/S{subject_id}/S{subject_id}.pkl"
                with open(subject_file, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                
                # Extract signals
                chest_signals = data['signal']['chest']
                wrist_signals = data['signal']['wrist']
                labels = data['label']
                
                # Combine chest and wrist data
                subject_data = {
                    'chest_eda': chest_signals['EDA'],
                    'chest_ecg': chest_signals['ECG'],
                    'chest_emg': chest_signals['EMG'],
                    'chest_temp': chest_signals['Temp'],
                    'chest_resp': chest_signals['Resp'],
                    'chest_acc': chest_signals['ACC'],
                    'wrist_eda': wrist_signals['EDA'],
                    'wrist_bvp': wrist_signals['BVP'],
                    'wrist_temp': wrist_signals['TEMP'],
                    'wrist_acc': wrist_signals['ACC'],
                    'labels': labels
                }
                
                all_data.append(subject_data)
                print(f"✅ Loaded subject S{subject_id}: {len(labels)} samples")
                
            except FileNotFoundError:
                print(f"⚠️ Subject S{subject_id} not found, skipping...")
                continue
        
        return all_data
    
    def extract_features(self, signal_data, fs=700, window_size=60):
        """Extract comprehensive features from physiological signals"""
        features = []
        
        for signal_name, signal_values in signal_data.items():
            if signal_name == 'labels':
                continue
                
            signal_array = np.array(signal_values).flatten()
            
            # Skip if signal is too short
            if len(signal_array) < window_size * fs:
                continue
            
            # Resample to common frequency if needed
            if len(signal_array) > window_size * fs:
                signal_array = signal_array[:window_size * fs]
            
            # Time domain features
            time_features = self._extract_time_features(signal_array)
            
            # Frequency domain features
            freq_features = self._extract_frequency_features(signal_array, fs)
            
            # Statistical features
            stat_features = self._extract_statistical_features(signal_array)
            
            # Combine all features
            signal_features = time_features + freq_features + stat_features
            features.extend(signal_features)
            
            # Store feature names (only once)
            if not self.feature_names:
                self.feature_names.extend([f"{signal_name}_{feat}" for feat in 
                    ['mean', 'std', 'var', 'min', 'max', 'range', 'skew', 'kurtosis',
                     'rms', 'peak_freq', 'spectral_centroid', 'spectral_bandwidth',
                     'spectral_rolloff', 'zero_crossing_rate', 'mfcc_1', 'mfcc_2']])
        
        return np.array(features)
    
    def _extract_time_features(self, signal):
        """Extract time domain features"""
        return [
            np.mean(signal),
            np.std(signal),
            np.var(signal),
            np.min(signal),
            np.max(signal),
            np.ptp(signal),  # peak-to-peak
            skew(signal),
            kurtosis(signal),
            np.sqrt(np.mean(signal**2))  # RMS
        ]
    
    def _extract_frequency_features(self, signal, fs):
        """Extract frequency domain features"""
        # Compute power spectral density
        freqs, psd = signal.welch(signal, fs=fs, nperseg=min(len(signal)//4, 1024))
        
        # Peak frequency
        peak_freq = freqs[np.argmax(psd)]
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
        
        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
        
        # Spectral rolloff (95% of energy)
        cumsum_psd = np.cumsum(psd)
        rolloff_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        
        # Zero crossing rate
        zero_crossing_rate = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
        
        # Simple MFCC-like features (first 2 coefficients)
        mfcc_1 = np.mean(np.abs(np.fft.fft(signal))[:len(signal)//4])
        mfcc_2 = np.std(np.abs(np.fft.fft(signal))[:len(signal)//4])
        
        return [peak_freq, spectral_centroid, spectral_bandwidth, 
                spectral_rolloff, zero_crossing_rate, mfcc_1, mfcc_2]
    
    def _extract_statistical_features(self, signal):
        """Extract additional statistical features"""
        return [
            np.median(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
            np.percentile(signal, 90),
            np.percentile(signal, 95)
        ]
    
    def preprocess_data(self, all_data, target_labels=[0, 1]):  # 0: baseline, 1: stress
        """Preprocess all data and extract features"""
        X = []
        y = []
        
        print("🔄 Preprocessing data and extracting features...")
        
        for subject_data in all_data:
            labels = subject_data['labels']
            
            # Create binary labels: 0 for baseline, 1 for stress/panic
            binary_labels = np.zeros_like(labels)
            binary_labels[labels == 1] = 1  # Stress = panic attack
            
            # Extract features for each window
            window_size = 60  # 60 seconds
            fs = 700  # WESAD sampling rate
            
            for i in range(0, len(labels) - window_size * fs, window_size * fs // 2):  # 50% overlap
                window_data = {}
                for key, value in subject_data.items():
                    if key != 'labels':
                        window_data[key] = value[i:i + window_size * fs]
                
                # Extract features
                features = self.extract_features(window_data, fs, window_size)
                
                if len(features) > 0:
                    X.append(features)
                    # Use majority label in window
                    window_labels = binary_labels[i:i + window_size * fs]
                    y.append(1 if np.sum(window_labels) > len(window_labels) * 0.5 else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Extracted {X.shape[0]} samples with {X.shape[1]} features")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple machine learning models"""
        print("🤖 Training machine learning models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
        
        self.models = results
        return X_test_scaled, y_test, results
    
    def plot_results(self, X_test, y_test, results):
        """Plot model results and feature importance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        aucs = [results[name]['auc'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        axes[0, 0].bar(x + width/2, aucs, width, label='AUC', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confusion matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_model = results[best_model_name]['model']
        y_pred = results[best_model_name]['predictions']
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. Feature importance (for tree-based models)
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            
            axes[1, 0].barh(range(len(indices)), importances[indices])
            axes[1, 0].set_yticks(range(len(indices)))
            axes[1, 0].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title(f'Top 20 Features - {best_model_name}')
        
        # 4. ROC curves
        for name, result in results.items():
            if result['probabilities'] is not None:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
                axes[1, 1].plot(fpr, tpr, label=f'{name} (AUC={result["auc"]:.3f})')
        
        axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\n" + "="*50)
        print("DETAILED MODEL RESULTS")
        print("="*50)
        
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  Accuracy: {result['accuracy']:.3f}")
            print(f"  AUC: {result['auc']:.3f}")
            
            # Classification report
            report = classification_report(y_test, result['predictions'], 
                                        target_names=['Baseline', 'Panic Attack'])
            print(f"  Classification Report:\n{report}")
    
    def predict_panic_attack(self, signal_data):
        """Predict panic attack for new data"""
        if not self.models:
            raise ValueError("No trained models available. Please train models first.")
        
        # Extract features
        features = self.extract_features(signal_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions from all models
        predictions = {}
        for name, result in self.models.items():
            model = result['model']
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0][1] if hasattr(model, 'predict_proba') else None
            predictions[name] = {'prediction': pred, 'probability': proba}
        
        return predictions

def main():
    """Main function to run the panic attack detection system"""
    print("🚨 Panic Attack Detection System")
    print("="*50)
    
    # Initialize detector
    detector = PanicAttackDetector()
    
    # Load data
    print("📁 Loading WESAD dataset...")
    all_data = detector.load_wesad_data(subject_ids=[2, 3, 4, 5, 6, 7, 8, 9, 10])  # Use subset for faster processing
    
    # Preprocess data
    X, y = detector.preprocess_data(all_data)
    
    # Train models
    X_test, y_test, results = detector.train_models(X, y)
    
    # Plot results
    detector.plot_results(X_test, y_test, results)
    
    print("\n✅ Panic attack detection system training completed!")
    print("The system is now ready to detect panic attacks in real-time physiological data.")

if __name__ == "__main__":
    main()
