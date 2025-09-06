import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class ClinicalThresholdDetector:
    """
    Clinical threshold-based panic attack detection system based on DSM-5 criteria
    and practical sensor thresholds for physiological monitoring.
    """
    
    def __init__(self):
        self.baseline_data = {}
        self.thresholds = {
            'heart_rate': {
                'sudden_increase': 30,  # bpm above baseline
                'absolute_threshold': 120,  # bpm at rest
                'window_size': 60  # seconds for baseline calculation
            },
            'hrv': {
                'drop_percentage': 0.3,  # 30% drop from baseline
                'window_size': 300  # 5 minutes for HRV calculation
            },
            'breathing_rate': {
                'threshold': 20,  # breaths per minute
                'normal_range': (12, 18)  # normal adult range
            },
            'eda': {
                'spike_threshold': 0.05,  # µS/sec increase
                'window_size': 10  # seconds for spike detection
            },
            'tremor': {
                'frequency_range': (6, 12),  # Hz for hand tremors
                'variance_threshold': 1.5  # multiplier of baseline variance
            },
            'skin_temp': {
                'drop_threshold': 0.5,  # °C drop from baseline
                'window_size': 60  # seconds for temperature monitoring
            }
        }
        self.dsm5_symptoms = [
            'heart_palpitations', 'sweating', 'trembling', 'shortness_breath',
            'choking', 'chest_pain', 'nausea', 'dizziness', 'chills_heat',
            'numbness', 'unreality', 'fear_control', 'fear_dying'
        ]
        
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
        """Calculate baseline values for each physiological parameter"""
        chest_signals = signal_data['signal']['chest']
        wrist_signals = signal_data['signal']['wrist']
        
        # Get baseline data (first window_size seconds, assuming label 0 is baseline)
        labels = signal_data['label']
        baseline_indices = np.where(labels == 0)[0]
        
        if len(baseline_indices) < window_size:
            baseline_indices = baseline_indices[:len(baseline_indices)]
        else:
            baseline_indices = baseline_indices[:window_size]
        
        baseline = {}
        
        # Heart Rate (from ECG)
        if 'ECG' in chest_signals:
            ecg_baseline = chest_signals['ECG'][baseline_indices]
            baseline['heart_rate'] = self._calculate_heart_rate_from_ecg(ecg_baseline)
        
        # Heart Rate Variability (from ECG)
        if 'ECG' in chest_signals:
            baseline['hrv'] = self._calculate_hrv(chest_signals['ECG'][baseline_indices])
        
        # Breathing Rate (from Respiration)
        if 'Resp' in chest_signals:
            resp_baseline = chest_signals['Resp'][baseline_indices]
            baseline['breathing_rate'] = self._calculate_breathing_rate(resp_baseline)
        
        # EDA (Electrodermal Activity)
        if 'EDA' in chest_signals:
            eda_baseline = chest_signals['EDA'][baseline_indices]
            baseline['eda'] = {
                'mean': np.mean(eda_baseline),
                'std': np.std(eda_baseline),
                'rate_of_change': self._calculate_eda_rate_of_change(eda_baseline)
            }
        
        # Tremor (from Accelerometer)
        if 'ACC' in wrist_signals:
            acc_baseline = wrist_signals['ACC'][baseline_indices]
            baseline['tremor'] = self._calculate_tremor_metrics(acc_baseline)
        
        # Skin Temperature
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
            # Find R-peaks using peak detection
            peaks, _ = signal.find_peaks(ecg_signal.flatten(), height=np.std(ecg_signal), distance=fs//3)
            
            if len(peaks) > 1:
                # Calculate RR intervals
                rr_intervals = np.diff(peaks) / fs
                # Calculate heart rate (beats per minute)
                heart_rate = 60 / np.mean(rr_intervals)
                return heart_rate
            else:
                return 70  # Default resting heart rate
        except:
            return 70
    
    def _calculate_hrv(self, ecg_signal, fs=700):
        """Calculate Heart Rate Variability (RMSSD)"""
        try:
            peaks, _ = signal.find_peaks(ecg_signal.flatten(), height=np.std(ecg_signal), distance=fs//3)
            
            if len(peaks) > 2:
                rr_intervals = np.diff(peaks) / fs
                # Calculate RMSSD (Root Mean Square of Successive Differences)
                rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
                return rmssd
            else:
                return 0.03  # Default RMSSD
        except:
            return 0.03
    
    def _calculate_breathing_rate(self, resp_signal, fs=700):
        """Calculate breathing rate from respiration signal"""
        try:
            # Find breathing peaks
            peaks, _ = signal.find_peaks(resp_signal.flatten(), height=np.std(resp_signal), distance=fs//2)
            
            if len(peaks) > 1:
                # Calculate breathing intervals
                breathing_intervals = np.diff(peaks) / fs
                # Calculate breathing rate (breaths per minute)
                breathing_rate = 60 / np.mean(breathing_intervals)
                return breathing_rate
            else:
                return 15  # Default breathing rate
        except:
            return 15
    
    def _calculate_eda_rate_of_change(self, eda_signal, window_size=10):
        """Calculate EDA rate of change (µS/sec)"""
        try:
            if len(eda_signal) < window_size:
                return 0
            
            # Calculate rate of change over windows
            rates = []
            for i in range(0, len(eda_signal) - window_size, window_size):
                window = eda_signal[i:i + window_size]
                rate = (window[-1] - window[0]) / window_size
                rates.append(rate)
            
            return np.mean(rates)
        except:
            return 0
    
    def _calculate_tremor_metrics(self, acc_signal):
        """Calculate tremor metrics from accelerometer data"""
        try:
            # Calculate motion variance
            motion_variance = np.var(acc_signal, axis=0)
            total_variance = np.sum(motion_variance)
            
            # Calculate frequency content in tremor range (6-12 Hz)
            freqs = np.fft.fftfreq(len(acc_signal), 1/32)  # 32 Hz sampling rate for ACC
            fft = np.fft.fft(acc_signal.flatten())
            
            tremor_freq_mask = (freqs >= 6) & (freqs <= 12)
            tremor_power = np.sum(np.abs(fft[tremor_freq_mask]) ** 2)
            
            return {
                'variance': total_variance,
                'tremor_power': tremor_power
            }
        except:
            return {'variance': 0, 'tremor_power': 0}
    
    def detect_panic_symptoms(self, signal_data, baseline, window_size=60):
        """Detect panic attack symptoms based on clinical thresholds"""
        chest_signals = signal_data['signal']['chest']
        wrist_signals = signal_data['signal']['wrist']
        labels = signal_data['label']
        
        # Process data in windows
        step_size = window_size // 2  # 50% overlap
        panic_detections = []
        
        for i in range(0, len(labels) - window_size, step_size):
            window_data = {
                'chest_ecg': chest_signals['ECG'][i:i + window_size],
                'chest_resp': chest_signals['Resp'][i:i + window_size],
                'chest_eda': chest_signals['EDA'][i:i + window_size],
                'wrist_acc': wrist_signals['ACC'][i:i + window_size],
                'wrist_temp': wrist_signals['TEMP'][i:i + window_size],
                'labels': labels[i:i + window_size]
            }
            
            # Detect symptoms
            symptoms = self._detect_symptoms_in_window(window_data, baseline)
            panic_detections.append(symptoms)
        
        return panic_detections
    
    def _detect_symptoms_in_window(self, window_data, baseline):
        """Detect panic symptoms in a specific time window"""
        symptoms = {
            'heart_palpitations': False,
            'sweating': False,
            'trembling': False,
            'shortness_breath': False,
            'choking': False,
            'chest_pain': False,
            'nausea': False,
            'dizziness': False,
            'chills_heat': False,
            'numbness': False,
            'unreality': False,
            'fear_control': False,
            'fear_dying': False
        }
        
        # 1. Heart Palpitations (Heart Rate)
        if 'chest_ecg' in window_data:
            current_hr = self._calculate_heart_rate_from_ecg(window_data['chest_ecg'])
            baseline_hr = baseline.get('heart_rate', 70)
            
            if (current_hr >= baseline_hr + self.thresholds['heart_rate']['sudden_increase'] or 
                current_hr >= self.thresholds['heart_rate']['absolute_threshold']):
                symptoms['heart_palpitations'] = True
        
        # 2. Sweating (EDA spike)
        if 'chest_eda' in window_data:
            current_eda_rate = self._calculate_eda_rate_of_change(window_data['chest_eda'])
            baseline_eda_rate = baseline.get('eda', {}).get('rate_of_change', 0)
            
            if current_eda_rate >= baseline_eda_rate + self.thresholds['eda']['spike_threshold']:
                symptoms['sweating'] = True
        
        # 3. Trembling (Accelerometer)
        if 'wrist_acc' in window_data:
            tremor_metrics = self._calculate_tremor_metrics(window_data['wrist_acc'])
            baseline_tremor = baseline.get('tremor', {'variance': 0, 'tremor_power': 0})
            
            if (tremor_metrics['variance'] >= baseline_tremor['variance'] * self.thresholds['tremor']['variance_threshold'] or
                tremor_metrics['tremor_power'] >= baseline_tremor['tremor_power'] * 2):
                symptoms['trembling'] = True
        
        # 4. Shortness of Breath (Breathing Rate)
        if 'chest_resp' in window_data:
            current_br = self._calculate_breathing_rate(window_data['chest_resp'])
            
            if current_br >= self.thresholds['breathing_rate']['threshold']:
                symptoms['shortness_breath'] = True
        
        # 5. Chills/Heat Sensations (Skin Temperature)
        if 'wrist_temp' in window_data:
            current_temp = np.mean(window_data['wrist_temp'])
            baseline_temp = baseline.get('skin_temp', {}).get('mean', 36.5)
            
            if current_temp <= baseline_temp - self.thresholds['skin_temp']['drop_threshold']:
                symptoms['chills_heat'] = True
        
        # Count total symptoms
        symptom_count = sum(symptoms.values())
        
        # DSM-5 criteria: ≥4 symptoms = panic attack
        panic_attack = symptom_count >= 4
        
        return {
            'symptoms': symptoms,
            'symptom_count': symptom_count,
            'panic_attack': panic_attack,
            'window_labels': window_data['labels']
        }
    
    def evaluate_threshold_detection(self, all_data):
        """Evaluate the threshold-based detection system"""
        print("🔍 Evaluating Clinical Threshold Detection System")
        print("="*60)
        
        all_detections = []
        all_labels = []
        
        for i, data in enumerate(all_data):
            print(f"Processing subject {i+1}/{len(all_data)}...")
            
            # Calculate baseline
            baseline = self.calculate_baseline(data)
            print(f"  Baseline - HR: {baseline.get('heart_rate', 0):.1f} bpm, "
                  f"BR: {baseline.get('breathing_rate', 0):.1f} bpm, "
                  f"EDA: {baseline.get('eda', {}).get('mean', 0):.3f} µS")
            
            # Detect panic symptoms
            detections = self.detect_panic_symptoms(data, baseline)
            
            for detection in detections:
                all_detections.append(detection['panic_attack'])
                # Use majority vote for ground truth
                window_labels = detection['window_labels']
                ground_truth = 1 if np.sum(window_labels == 1) > len(window_labels) * 0.5 else 0
                all_labels.append(ground_truth)
        
        # Calculate metrics
        all_detections = np.array(all_detections)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_detections)
        
        print(f"\n📊 Threshold Detection Results:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Total samples: {len(all_detections)}")
        print(f"  Panic attacks detected: {np.sum(all_detections)}")
        print(f"  True panic attacks: {np.sum(all_labels)}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_detections)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives: {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives: {cm[1,1]}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(all_labels, all_detections, 
                                  target_names=['Baseline', 'Panic Attack']))
        
        return all_detections, all_labels, accuracy
    
    def plot_threshold_analysis(self, detections, labels):
        """Plot analysis of threshold-based detection"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(labels, detections)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Baseline', 'Panic Attack'],
                   yticklabels=['Baseline', 'Panic Attack'])
        axes[0, 0].set_title('Confusion Matrix - Clinical Thresholds')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Detection Timeline
        axes[0, 1].plot(detections[:500], label='Detected', alpha=0.7)
        axes[0, 1].plot(labels[:500], label='Ground Truth', alpha=0.7)
        axes[0, 1].set_title('Detection Timeline (First 500 samples)')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Panic Attack (1) / Baseline (0)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Class Distribution
        class_counts = np.bincount(labels)
        axes[1, 0].pie(class_counts, labels=['Baseline', 'Panic Attack'], 
                      autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        axes[1, 0].set_title('Ground Truth Class Distribution')
        
        # 4. Detection Accuracy by Class
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = ['Sensitivity', 'Specificity', 'Accuracy']
        values = [sensitivity, specificity, accuracy_score(labels, detections)]
        
        bars = axes[1, 1].bar(metrics, values, color=['lightgreen', 'lightblue', 'lightcoral'])
        axes[1, 1].set_title('Detection Performance Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def print_clinical_summary(self):
        """Print clinical summary of the detection system"""
        print("\n" + "="*60)
        print("🏥 CLINICAL THRESHOLD DETECTION SYSTEM SUMMARY")
        print("="*60)
        print("Based on DSM-5 criteria for panic attack diagnosis")
        print("\n📋 Detected Symptoms:")
        print("  ✅ Heart Palpitations (HR ≥30 bpm above baseline OR >120 bpm)")
        print("  ✅ Sweating (EDA spike >0.05 µS/sec)")
        print("  ✅ Trembling (Increased accelerometer variance)")
        print("  ✅ Shortness of Breath (Breathing rate >20 bpm)")
        print("  ✅ Chills/Heat (Skin temp drop >0.5°C)")
        print("\n🎯 DSM-5 Criteria:")
        print("  • Panic attack = ≥4 symptoms appearing suddenly")
        print("  • Symptoms must peak within minutes")
        print("  • Detection window: 60 seconds with 50% overlap")
        print("\n⚙️ Thresholds (Personalized per individual):")
        for param, thresholds in self.thresholds.items():
            print(f"  • {param.replace('_', ' ').title()}: {thresholds}")

def main():
    """Main function to run the clinical threshold detection system"""
    print("🏥 Clinical Threshold-Based Panic Attack Detection System")
    print("Based on DSM-5 criteria and practical sensor thresholds")
    print("="*70)
    
    # Initialize detector
    detector = ClinicalThresholdDetector()
    
    # Load data
    print("📁 Loading WESAD dataset...")
    all_data = detector.load_wesad_data(subject_ids=[2, 3, 4, 5, 6])  # Use subset for faster processing
    
    if not all_data:
        print("❌ No data loaded. Exiting...")
        return
    
    # Evaluate threshold detection
    detections, labels, accuracy = detector.evaluate_threshold_detection(all_data)
    
    # Plot results
    detector.plot_threshold_analysis(detections, labels)
    
    # Print clinical summary
    detector.print_clinical_summary()
    
    print(f"\n✅ Clinical threshold detection system evaluation completed!")
    print(f"🎯 Final Accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()
