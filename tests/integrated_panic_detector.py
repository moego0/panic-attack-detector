import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class IntegratedPanicDetector:
    """
    Integrated ML model that combines multiple approaches for panic attack detection:
    1. Clinical threshold-based features
    2. Comprehensive statistical features
    3. Ensemble learning with multiple algorithms
    4. Real-time prediction capability
    """
    
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_selector = SelectKBest(f_classif, k=50)  # Select top 50 features
        self.ensemble_model = None
        self.individual_models = {}
        self.feature_names = []
        self.baseline_data = {}
        self.thresholds = {
            'heart_rate': {'sudden_increase': 25, 'absolute_threshold': 110},
            'eda': {'spike_threshold': 0.03},
            'breathing_rate': {'threshold': 18},
            'tremor': {'variance_threshold': 1.3},
            'skin_temp': {'drop_threshold': 0.3}
        }
        
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
    
    def calculate_baseline(self, signal_data, window_size=300):
        """Calculate personalized baseline for each subject"""
        chest_signals = signal_data['signal']['chest']
        wrist_signals = signal_data['signal']['wrist']
        labels = signal_data['label']
        
        # Get baseline data (label 0)
        baseline_indices = np.where(labels == 0)[0]
        if len(baseline_indices) < window_size:
            baseline_indices = baseline_indices[:len(baseline_indices)]
        else:
            baseline_indices = baseline_indices[:window_size]
        
        baseline = {}
        
        # Heart Rate baseline
        if 'ECG' in chest_signals:
            ecg_baseline = chest_signals['ECG'][baseline_indices]
            baseline['heart_rate'] = self._calculate_heart_rate_from_ecg(ecg_baseline)
        
        # EDA baseline
        if 'EDA' in chest_signals:
            eda_baseline = chest_signals['EDA'][baseline_indices]
            baseline['eda'] = {
                'mean': np.mean(eda_baseline),
                'std': np.std(eda_baseline),
                'rate_of_change': self._calculate_eda_rate_of_change(eda_baseline)
            }
        
        # Breathing rate baseline
        if 'Resp' in chest_signals:
            resp_baseline = chest_signals['Resp'][baseline_indices]
            baseline['breathing_rate'] = self._calculate_breathing_rate(resp_baseline)
        
        # Tremor baseline
        if 'ACC' in wrist_signals:
            acc_baseline = wrist_signals['ACC'][baseline_indices]
            baseline['tremor'] = self._calculate_tremor_metrics(acc_baseline)
        
        # Temperature baseline
        if 'TEMP' in wrist_signals:
            temp_baseline = wrist_signals['TEMP'][baseline_indices]
            baseline['skin_temp'] = {
                'mean': np.mean(temp_baseline),
                'std': np.std(temp_baseline)
            }
        
        return baseline
    
    def _calculate_heart_rate_from_ecg(self, ecg_signal, fs=700):
        """Calculate heart rate from ECG signal"""
        try:
            peaks, _ = signal.find_peaks(ecg_signal.flatten(), height=np.std(ecg_signal), distance=fs//3)
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / fs
                return 60 / np.mean(rr_intervals)
            return 70
        except:
            return 70
    
    def _calculate_eda_rate_of_change(self, eda_signal, window_size=10):
        """Calculate EDA rate of change"""
        try:
            if len(eda_signal) < window_size:
                return 0
            rates = []
            for i in range(0, len(eda_signal) - window_size, window_size):
                window = eda_signal[i:i + window_size]
                rate = (window[-1] - window[0]) / window_size
                rates.append(rate)
            return np.mean(rates)
        except:
            return 0
    
    def _calculate_breathing_rate(self, resp_signal, fs=700):
        """Calculate breathing rate from respiration signal"""
        try:
            peaks, _ = signal.find_peaks(resp_signal.flatten(), height=np.std(resp_signal), distance=fs//2)
            if len(peaks) > 1:
                breathing_intervals = np.diff(peaks) / fs
                return 60 / np.mean(breathing_intervals)
            return 15
        except:
            return 15
    
    def _calculate_tremor_metrics(self, acc_signal):
        """Calculate tremor metrics from accelerometer"""
        try:
            motion_variance = np.var(acc_signal, axis=0)
            total_variance = np.sum(motion_variance)
            
            freqs = np.fft.fftfreq(len(acc_signal), 1/32)
            fft = np.fft.fft(acc_signal.flatten())
            tremor_freq_mask = (freqs >= 6) & (freqs <= 12)
            tremor_power = np.sum(np.abs(fft[tremor_freq_mask]) ** 2)
            
            return {'variance': total_variance, 'tremor_power': tremor_power}
        except:
            return {'variance': 0, 'tremor_power': 0}
    
    def extract_integrated_features(self, signal_data, baseline, window_size=7000, overlap=0.5):
        """Extract integrated features combining clinical and statistical approaches"""
        features = []
        labels = []
        
        chest_signals = signal_data['signal']['chest']
        wrist_signals = signal_data['signal']['wrist']
        label_data = signal_data['label']
        
        step_size = int(window_size * (1 - overlap))
        
        for i in range(0, len(label_data) - window_size, step_size):
            try:
                window_features = []
                
                # Get window data
                window_data = {
                    'chest_ecg': chest_signals['ECG'][i:i + window_size],
                    'chest_resp': chest_signals['Resp'][i:i + window_size],
                    'chest_eda': chest_signals['EDA'][i:i + window_size],
                    'chest_emg': chest_signals['EMG'][i:i + window_size],
                    'chest_temp': chest_signals['Temp'][i:i + window_size],
                    'wrist_eda': wrist_signals['EDA'][i:i + window_size],
                    'wrist_bvp': wrist_signals['BVP'][i:i + window_size],
                    'wrist_temp': wrist_signals['TEMP'][i:i + window_size],
                    'wrist_acc': wrist_signals['ACC'][i:i + window_size]
                }
                
                # 1. CLINICAL THRESHOLD FEATURES (5 features)
                clinical_features = self._extract_clinical_features(window_data, baseline)
                window_features.extend(clinical_features)
                
                # 2. STATISTICAL FEATURES (per signal)
                for signal_name, signal_data in window_data.items():
                    if len(signal_data) > 0:
                        stat_features = self._extract_statistical_features(signal_data.flatten())
                        window_features.extend(stat_features)
                    else:
                        window_features.extend([0] * 8)  # 8 features per signal
                
                # 3. FREQUENCY DOMAIN FEATURES (per signal)
                for signal_name, signal_data in window_data.items():
                    if len(signal_data) > 0:
                        freq_features = self._extract_frequency_features(signal_data.flatten(), fs=700)
                        window_features.extend(freq_features)
                    else:
                        window_features.extend([0] * 5)  # 5 features per signal
                
                # 4. TIME SERIES FEATURES (per signal)
                for signal_name, signal_data in window_data.items():
                    if len(signal_data) > 0:
                        time_features = self._extract_time_series_features(signal_data.flatten())
                        window_features.extend(time_features)
                    else:
                        window_features.extend([0] * 6)  # 6 features per signal
                
                # 5. CROSS-SIGNAL FEATURES (correlations between signals)
                cross_features = self._extract_cross_signal_features(window_data)
                window_features.extend(cross_features)
                
                # Get label (majority vote)
                window_labels = label_data[i:i + window_size]
                binary_label = 1 if np.sum(window_labels == 1) > len(window_labels) * 0.5 else 0
                
                features.append(window_features)
                labels.append(binary_label)
                
            except Exception as e:
                print(f"Error processing window {i}: {e}")
                continue
        
        return np.array(features), np.array(labels)
    
    def _extract_clinical_features(self, window_data, baseline):
        """Extract clinical threshold-based features"""
        features = []
        
        # Heart rate features
        if 'chest_ecg' in window_data:
            current_hr = self._calculate_heart_rate_from_ecg(window_data['chest_ecg'])
            baseline_hr = baseline.get('heart_rate', 70)
            features.extend([
                current_hr,
                current_hr - baseline_hr,
                1 if current_hr >= baseline_hr + self.thresholds['heart_rate']['sudden_increase'] else 0,
                1 if current_hr >= self.thresholds['heart_rate']['absolute_threshold'] else 0
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # EDA features
        if 'chest_eda' in window_data:
            current_eda_rate = self._calculate_eda_rate_of_change(window_data['chest_eda'])
            baseline_eda_rate = baseline.get('eda', {}).get('rate_of_change', 0)
            features.extend([
                current_eda_rate,
                1 if current_eda_rate >= baseline_eda_rate + self.thresholds['eda']['spike_threshold'] else 0
            ])
        else:
            features.extend([0, 0])
        
        # Breathing rate features
        if 'chest_resp' in window_data:
            current_br = self._calculate_breathing_rate(window_data['chest_resp'])
            features.extend([
                current_br,
                1 if current_br >= self.thresholds['breathing_rate']['threshold'] else 0
            ])
        else:
            features.extend([0, 0])
        
        # Tremor features
        if 'wrist_acc' in window_data:
            tremor_metrics = self._calculate_tremor_metrics(window_data['wrist_acc'])
            baseline_tremor = baseline.get('tremor', {'variance': 0, 'tremor_power': 0})
            features.extend([
                tremor_metrics['variance'],
                1 if tremor_metrics['variance'] >= baseline_tremor['variance'] * self.thresholds['tremor']['variance_threshold'] else 0
            ])
        else:
            features.extend([0, 0])
        
        # Temperature features
        if 'wrist_temp' in window_data:
            current_temp = np.mean(window_data['wrist_temp'])
            baseline_temp = baseline.get('skin_temp', {}).get('mean', 36.5)
            features.extend([
                current_temp,
                1 if current_temp <= baseline_temp - self.thresholds['skin_temp']['drop_threshold'] else 0
            ])
        else:
            features.extend([0, 0])
        
        return features
    
    def _extract_statistical_features(self, signal):
        """Extract statistical features from signal"""
        if len(signal) == 0:
            return [0] * 8
        
        try:
            # Clean signal data
            signal_clean = signal[np.isfinite(signal)]
            if len(signal_clean) == 0:
                return [0] * 8
            
            features = [
                np.mean(signal_clean),
                np.std(signal_clean),
                np.var(signal_clean),
                np.min(signal_clean),
                np.max(signal_clean),
                np.ptp(signal_clean),  # peak-to-peak
                skew(signal_clean) if len(signal_clean) > 2 else 0,
                kurtosis(signal_clean) if len(signal_clean) > 2 else 0
            ]
            
            # Replace any NaN or infinite values with 0
            features = [0 if not np.isfinite(f) else f for f in features]
            return features
        except:
            return [0] * 8
    
    def _extract_frequency_features(self, signal, fs=700):
        """Extract frequency domain features"""
        if len(signal) == 0:
            return [0] * 5
        
        try:
            # Clean signal data
            signal_clean = signal[np.isfinite(signal)]
            if len(signal_clean) < 4:
                return [0] * 5
            
            freqs, psd = signal.welch(signal_clean, fs=fs, nperseg=min(len(signal_clean)//4, 1024))
            
            peak_freq = freqs[np.argmax(psd)]
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0
            
            cumsum_psd = np.cumsum(psd)
            rolloff_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
            
            zero_crossing_rate = np.sum(np.diff(np.sign(signal_clean)) != 0) / len(signal_clean)
            
            features = [peak_freq, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate]
            features = [0 if not np.isfinite(f) else f for f in features]
            return features
        except:
            return [0] * 5
    
    def _extract_time_series_features(self, signal):
        """Extract time series features"""
        if len(signal) == 0:
            return [0] * 6
        
        try:
            # Clean signal data
            signal_clean = signal[np.isfinite(signal)]
            if len(signal_clean) == 0:
                return [0] * 6
            
            # Trend features
            x = np.arange(len(signal_clean))
            slope = np.polyfit(x, signal_clean, 1)[0] if len(signal_clean) > 1 else 0
            
            # Autocorrelation
            autocorr = np.corrcoef(signal_clean[:-1], signal_clean[1:])[0, 1] if len(signal_clean) > 1 else 0
            
            # Energy
            energy = np.sum(signal_clean ** 2)
            
            # Entropy (approximate)
            hist, _ = np.histogram(signal_clean, bins=10)
            hist = hist / np.sum(hist)
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(signal_clean)) != 0)
            
            # Mean absolute deviation
            mad = np.mean(np.abs(signal_clean - np.mean(signal_clean)))
            
            features = [slope, autocorr, energy, entropy, zero_crossings, mad]
            features = [0 if not np.isfinite(f) else f for f in features]
            return features
        except:
            return [0] * 6
    
    def _extract_cross_signal_features(self, window_data):
        """Extract cross-signal correlation features"""
        features = []
        
        # Get available signals
        signals = {}
        for name, data in window_data.items():
            if len(data) > 0:
                signals[name] = data.flatten()
        
        # Calculate correlations between different signal types
        signal_pairs = [
            ('chest_ecg', 'chest_resp'),
            ('chest_eda', 'wrist_eda'),
            ('chest_temp', 'wrist_temp'),
            ('wrist_bvp', 'chest_ecg'),
            ('wrist_acc', 'chest_emg')
        ]
        
        for sig1, sig2 in signal_pairs:
            if sig1 in signals and sig2 in signals:
                try:
                    # Ensure same length
                    min_len = min(len(signals[sig1]), len(signals[sig2]))
                    corr = np.corrcoef(signals[sig1][:min_len], signals[sig2][:min_len])[0, 1]
                    features.append(corr if not np.isnan(corr) else 0)
                except:
                    features.append(0)
            else:
                features.append(0)
        
        return features
    
    def train_integrated_model(self, X, y):
        """Train the integrated ensemble model"""
        print("🤖 Training Integrated Panic Attack Detection Model...")
        
        # Clean data - handle NaN and infinite values
        print("🧹 Cleaning data...")
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Remove any remaining NaN or infinite values
        valid_mask = np.isfinite(X_clean).all(axis=1)
        X_clean = X_clean[valid_mask]
        y_clean = y[valid_mask]
        
        print(f"  Removed {len(y) - len(y_clean)} invalid samples")
        print(f"  Final dataset: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        print(f"Selected {X_train_selected.shape[1]} features from {X_train_scaled.shape[1]} total features")
        
        # Train individual models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=8, random_state=42),
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=0.1),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50, 25), random_state=42, max_iter=1000, alpha=0.01)
        }
        
        # Train and evaluate individual models
        individual_results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_selected, y_train)
            
            y_pred = model.predict(X_test_selected)
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            individual_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  {name} - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
        
        # Create ensemble model
        ensemble_models = [
            ('rf', individual_results['Random Forest']['model']),
            ('gb', individual_results['Gradient Boosting']['model']),
            ('svm', individual_results['SVM']['model']),
            ('lr', individual_results['Logistic Regression']['model']),
            ('nn', individual_results['Neural Network']['model'])
        ]
        
        self.ensemble_model = VotingClassifier(estimators=ensemble_models, voting='soft')
        self.ensemble_model.fit(X_train_selected, y_train)
        
        # Evaluate ensemble
        ensemble_pred = self.ensemble_model.predict(X_test_selected)
        ensemble_proba = self.ensemble_model.predict_proba(X_test_selected)[:, 1]
        
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)
        
        print(f"\n🎯 Ensemble Model - Accuracy: {ensemble_accuracy:.3f}, AUC: {ensemble_auc:.3f}")
        
        self.individual_models = individual_results
        
        return X_test_selected, y_test, {
            'ensemble': {
                'accuracy': ensemble_accuracy,
                'auc': ensemble_auc,
                'predictions': ensemble_pred,
                'probabilities': ensemble_proba
            },
            'individual': individual_results
        }
    
    def plot_integrated_results(self, X_test, y_test, results):
        """Plot comprehensive results for the integrated model"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Model comparison
        model_names = list(results['individual'].keys()) + ['Ensemble']
        accuracies = [results['individual'][name]['accuracy'] for name in results['individual'].keys()] + [results['ensemble']['accuracy']]
        aucs = [results['individual'][name]['auc'] for name in results['individual'].keys()] + [results['ensemble']['auc']]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, aucs, width, label='AUC', alpha=0.8, color='lightcoral')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Integrated Model Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Ensemble confusion matrix
        cm = confusion_matrix(y_test, results['ensemble']['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                   xticklabels=['Baseline', 'Panic Attack'],
                   yticklabels=['Baseline', 'Panic Attack'])
        axes[0, 1].set_title('Ensemble Model Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. ROC curves
        for name, result in results['individual'].items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            axes[0, 2].plot(fpr, tpr, label=f'{name} (AUC={result["auc"]:.3f})', linewidth=2)
        
        # Add ensemble ROC curve
        fpr, tpr, _ = roc_curve(y_test, results['ensemble']['probabilities'])
        axes[0, 2].plot(fpr, tpr, label=f'Ensemble (AUC={results["ensemble"]["auc"]:.3f})', 
                       linewidth=3, color='red', linestyle='--')
        
        axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        axes[0, 2].set_xlabel('False Positive Rate')
        axes[0, 2].set_ylabel('True Positive Rate')
        axes[0, 2].set_title('ROC Curves Comparison')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Feature importance (Random Forest)
        if hasattr(results['individual']['Random Forest']['model'], 'feature_importances_'):
            importances = results['individual']['Random Forest']['model'].feature_importances_
            indices = np.argsort(importances)[::-1][:15]
            
            axes[1, 0].barh(range(len(indices)), importances[indices])
            axes[1, 0].set_yticks(range(len(indices)))
            axes[1, 0].set_yticklabels([f'Feature_{i}' for i in indices])
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title('Top 15 Features (Random Forest)')
        
        # 5. Class distribution
        class_counts = np.bincount(y_test)
        axes[1, 1].pie(class_counts, labels=['Baseline', 'Panic Attack'], 
                      autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        axes[1, 1].set_title('Test Set Class Distribution')
        
        # 6. Performance metrics comparison
        metrics_data = []
        for name, result in results['individual'].items():
            metrics_data.append([name, result['accuracy'], result['auc']])
        metrics_data.append(['Ensemble', results['ensemble']['accuracy'], results['ensemble']['auc']])
        
        metrics_df = pd.DataFrame(metrics_data, columns=['Model', 'Accuracy', 'AUC'])
        metrics_df = metrics_df.sort_values('Accuracy', ascending=True)
        
        axes[1, 2].barh(metrics_df['Model'], metrics_df['Accuracy'], color='lightgreen', alpha=0.7)
        axes[1, 2].set_xlabel('Accuracy')
        axes[1, 2].set_title('Model Accuracy Ranking')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\n" + "="*70)
        print("INTEGRATED PANIC ATTACK DETECTION SYSTEM RESULTS")
        print("="*70)
        
        print(f"\n🎯 Ensemble Model Performance:")
        print(f"  Accuracy: {results['ensemble']['accuracy']:.3f}")
        print(f"  AUC: {results['ensemble']['auc']:.3f}")
        
        print(f"\n📊 Individual Model Performance:")
        for name, result in results['individual'].items():
            print(f"  {name}: Accuracy={result['accuracy']:.3f}, AUC={result['auc']:.3f}")
        
        print(f"\n📋 Classification Report (Ensemble):")
        print(classification_report(y_test, results['ensemble']['predictions'], 
                                  target_names=['Baseline', 'Panic Attack']))
    
    def predict_panic_attack(self, signal_data, baseline):
        """Predict panic attack for new data using the integrated model"""
        if self.ensemble_model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Extract features
        features, _ = self.extract_integrated_features(signal_data, baseline, window_size=7000)
        
        if len(features) == 0:
            return {'prediction': 0, 'probability': 0.0, 'confidence': 'low'}
        
        # Clean features - handle NaN and infinite values
        features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale and select features
        features_scaled = self.scaler.transform(features_clean)
        features_selected = self.feature_selector.transform(features_scaled)
        
        # Get predictions
        prediction = self.ensemble_model.predict(features_selected)[0]
        probability = self.ensemble_model.predict_proba(features_selected)[0][1]
        
        # Determine confidence level
        if probability > 0.8 or probability < 0.2:
            confidence = 'high'
        elif probability > 0.6 or probability < 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence': confidence,
            'interpretation': 'Panic Attack Detected' if prediction == 1 else 'Normal State'
        }

def main():
    """Main function to run the integrated panic attack detection system"""
    print("🚨 Integrated Panic Attack Detection System")
    print("Combining Clinical Thresholds + Advanced ML + Ensemble Learning")
    print("="*80)
    
    # Initialize detector
    detector = IntegratedPanicDetector()
    
    # Load data
    print("📁 Loading WESAD dataset...")
    all_data = detector.load_wesad_data(subject_ids=[2, 3, 4, 5, 6])  # Use subset for faster processing
    
    if not all_data:
        print("❌ No data loaded. Exiting...")
        return
    
    # Extract integrated features from all subjects
    print("🔄 Extracting integrated features...")
    all_features = []
    all_labels = []
    
    for i, data in enumerate(all_data):
        print(f"Processing subject {i+1}/{len(all_data)}...")
        
        # Calculate baseline
        baseline = detector.calculate_baseline(data)
        detector.baseline_data[f'subject_{i}'] = baseline
        
        # Extract features
        features, labels = detector.extract_integrated_features(data, baseline)
        all_features.append(features)
        all_labels.append(labels)
    
    # Combine all data
    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    
    print(f"✅ Extracted {X.shape[0]} samples with {X.shape[1]} features")
    print(f"📊 Class distribution: {np.bincount(y)}")
    class_percentages = np.bincount(y) / len(y) * 100
    print(f"📊 Class percentages: {class_percentages[0]:.1f}% Baseline, {class_percentages[1]:.1f}% Panic Attack")
    
    # Train integrated model
    X_test, y_test, results = detector.train_integrated_model(X, y)
    
    # Plot results
    detector.plot_integrated_results(X_test, y_test, results)
    
    print("\n✅ Integrated panic attack detection system training completed!")
    print("The system is now ready for real-time panic attack detection.")
    
    # Example prediction
    print("\n🔮 Example Prediction:")
    example_data = all_data[0]  # Use first subject as example
    example_baseline = detector.baseline_data['subject_0']
    prediction = detector.predict_panic_attack(example_data, example_baseline)
    print(f"  Prediction: {prediction['interpretation']}")
    print(f"  Probability: {prediction['probability']:.3f}")
    print(f"  Confidence: {prediction['confidence']}")

if __name__ == "__main__":
    main()
