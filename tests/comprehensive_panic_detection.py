import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

class ComprehensivePanicDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        
    def load_wesad_data(self, subject_ids=None):
        """Load WESAD dataset for specified subjects"""
        if subject_ids is None:
            subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        all_data = []
        
        for subject_id in subject_ids:
            try:
                subject_file = f"WESAD/S{subject_id}/S{subject_id}.pkl"
                with open(subject_file, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                
                print(f"✅ Loaded subject S{subject_id}: {len(data['label'])} samples")
                all_data.append(data)
                
            except FileNotFoundError:
                print(f"⚠️ Subject S{subject_id} not found, skipping...")
                continue
        
        return all_data
    
    def extract_comprehensive_features(self, signal_data, window_size=7000, overlap=0.5):
        """Extract comprehensive features from physiological signals"""
        features = []
        labels = []
        
        # Get all signals
        chest_signals = signal_data['signal']['chest']
        wrist_signals = signal_data['signal']['wrist']
        label_data = signal_data['label']
        
        # Calculate step size for overlapping windows
        step_size = int(window_size * (1 - overlap))
        
        # Process in overlapping windows
        for i in range(0, len(label_data) - window_size, step_size):
            try:
                window_features = []
                
                # Process each signal type
                signal_types = {
                    'chest_eda': chest_signals['EDA'],
                    'chest_ecg': chest_signals['ECG'],
                    'chest_emg': chest_signals['EMG'],
                    'chest_temp': chest_signals['Temp'],
                    'chest_resp': chest_signals['Resp'],
                    'wrist_eda': wrist_signals['EDA'],
                    'wrist_bvp': wrist_signals['BVP'],
                    'wrist_temp': wrist_signals['TEMP']
                }
                
                for signal_name, signal_data in signal_types.items():
                    signal_window = signal_data[i:i + window_size]
                    
                    if len(signal_window) > 0:
                        # Time domain features
                        time_features = self._extract_time_domain_features(signal_window.flatten())
                        window_features.extend(time_features)
                        
                        # Frequency domain features
                        freq_features = self._extract_frequency_domain_features(signal_window.flatten(), fs=700)
                        window_features.extend(freq_features)
                        
                        # Statistical features
                        stat_features = self._extract_statistical_features(signal_window.flatten())
                        window_features.extend(stat_features)
                    else:
                        # Add zeros if signal is empty
                        window_features.extend([0] * 20)  # 20 features per signal
                
                # Get label for this window (majority vote)
                window_labels = label_data[i:i + window_size]
                binary_label = 1 if np.sum(window_labels == 1) > len(window_labels) * 0.5 else 0
                
                features.append(window_features)
                labels.append(binary_label)
                
            except Exception as e:
                print(f"Error processing window {i}: {e}")
                continue
        
        return np.array(features), np.array(labels)
    
    def _extract_time_domain_features(self, signal):
        """Extract time domain features"""
        if len(signal) == 0:
            return [0] * 5
        
        return [
            np.mean(signal),
            np.std(signal),
            np.var(signal),
            np.max(signal) - np.min(signal),  # Range
            np.sqrt(np.mean(signal**2))  # RMS
        ]
    
    def _extract_frequency_domain_features(self, signal, fs=700):
        """Extract frequency domain features"""
        if len(signal) == 0:
            return [0] * 5
        
        try:
            # Compute power spectral density
            freqs, psd = signal.welch(signal, fs=fs, nperseg=min(len(signal)//4, 1024))
            
            # Peak frequency
            peak_freq = freqs[np.argmax(psd)]
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
            
            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0
            
            # Spectral rolloff (95% of energy)
            cumsum_psd = np.cumsum(psd)
            rolloff_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
            
            # Zero crossing rate
            zero_crossing_rate = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
            
            return [peak_freq, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate]
            
        except:
            return [0] * 5
    
    def _extract_statistical_features(self, signal):
        """Extract statistical features"""
        if len(signal) == 0:
            return [0] * 10
        
        try:
            return [
                np.median(signal),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
                np.percentile(signal, 90),
                np.percentile(signal, 95),
                skew(signal),
                kurtosis(signal),
                np.mean(np.abs(signal)),  # Mean absolute value
                np.sum(signal > np.mean(signal)) / len(signal),  # Above mean ratio
                np.sum(signal < np.mean(signal)) / len(signal)   # Below mean ratio
            ]
        except:
            return [0] * 10
    
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
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
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
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
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
        """Plot comprehensive results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Model comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        aucs = [results[name]['auc'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, aucs, width, label='AUC', alpha=0.8, color='lightcoral')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confusion matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        y_pred = results[best_model_name]['predictions']
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                   xticklabels=['Baseline', 'Panic Attack'],
                   yticklabels=['Baseline', 'Panic Attack'])
        axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. ROC curves
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            axes[0, 2].plot(fpr, tpr, label=f'{name} (AUC={result["auc"]:.3f})', linewidth=2)
        
        axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        axes[0, 2].set_xlabel('False Positive Rate')
        axes[0, 2].set_ylabel('True Positive Rate')
        axes[0, 2].set_title('ROC Curves')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Feature importance (for Random Forest)
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # Top 15 features
            
            axes[1, 0].barh(range(len(indices)), importances[indices])
            axes[1, 0].set_yticks(range(len(indices)))
            axes[1, 0].set_yticklabels([f'Feature_{i}' for i in indices])
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title('Top 15 Features (Random Forest)')
        
        # 5. Class distribution
        class_counts = np.bincount(y_test)
        axes[1, 1].pie(class_counts, labels=['Baseline', 'Panic Attack'], autopct='%1.1f%%', 
                      colors=['lightblue', 'lightcoral'])
        axes[1, 1].set_title('Test Set Class Distribution')
        
        # 6. Model performance summary
        performance_data = []
        for name, result in results.items():
            performance_data.append([name, result['accuracy'], result['auc']])
        
        performance_df = pd.DataFrame(performance_data, columns=['Model', 'Accuracy', 'AUC'])
        performance_df = performance_df.sort_values('Accuracy', ascending=True)
        
        axes[1, 2].barh(performance_df['Model'], performance_df['Accuracy'], color='lightgreen', alpha=0.7)
        axes[1, 2].set_xlabel('Accuracy')
        axes[1, 2].set_title('Model Accuracy Ranking')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\n" + "="*60)
        print("DETAILED MODEL RESULTS")
        print("="*60)
        
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  Accuracy: {result['accuracy']:.3f}")
            print(f"  AUC: {result['auc']:.3f}")
            
            # Classification report
            report = classification_report(y_test, result['predictions'], 
                                        target_names=['Baseline', 'Panic Attack'])
            print(f"  Classification Report:\n{report}")

def main():
    """Main function to run the comprehensive panic attack detection system"""
    print("🚨 Comprehensive Panic Attack Detection System")
    print("="*60)
    
    # Initialize detector
    detector = ComprehensivePanicDetector()
    
    # Load data
    print("📁 Loading WESAD dataset...")
    all_data = detector.load_wesad_data(subject_ids=[2, 3, 4, 5, 6])  # Use subset for faster processing
    
    if not all_data:
        print("❌ No data loaded. Exiting...")
        return
    
    # Extract features from all subjects
    print("🔄 Extracting comprehensive features...")
    all_features = []
    all_labels = []
    
    for i, data in enumerate(all_data):
        print(f"Processing subject {i+1}/{len(all_data)}...")
        features, labels = detector.extract_comprehensive_features(data)
        all_features.append(features)
        all_labels.append(labels)
    
    # Combine all data
    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    
    print(f"✅ Extracted {X.shape[0]} samples with {X.shape[1]} features")
    print(f"📊 Class distribution: {np.bincount(y)}")
    class_percentages = np.bincount(y) / len(y) * 100
    print(f"📊 Class percentages: {class_percentages[0]:.1f}% Baseline, {class_percentages[1]:.1f}% Panic Attack")
    
    # Train models
    X_test, y_test, results = detector.train_models(X, y)
    
    # Plot results
    detector.plot_results(X_test, y_test, results)
    
    print("\n✅ Comprehensive panic attack detection system training completed!")
    print("The system is now ready to detect panic attacks in real-time physiological data.")

if __name__ == "__main__":
    main()
