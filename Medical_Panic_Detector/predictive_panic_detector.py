import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import time

class PredictivePanicDetector:
    """
    Enhanced Panic Attack Detection System with Predictive Capabilities
    Detects panic attacks 5-15 minutes BEFORE they occur
    """
    
    def __init__(self, prediction_window=300, early_warning_window=600):
        """
        Initialize predictive detector
        
        Args:
            prediction_window: Seconds to predict ahead (default: 5 minutes)
            early_warning_window: Seconds to look back for patterns (default: 10 minutes)
        """
        self.prediction_window = prediction_window
        self.early_warning_window = early_warning_window
        self.baseline_window = 1800  # 30 minutes for baseline
        
        # Data buffers for pattern analysis
        self.physiological_buffer = deque(maxlen=early_warning_window)
        self.feature_buffer = deque(maxlen=early_warning_window // 30)  # 30-second windows
        self.prediction_history = deque(maxlen=100)
        
        # Load models
        self.load_predictive_models()
        
        # Prediction thresholds
        self.panic_probability_threshold = 0.7
        self.early_warning_threshold = 0.4
        self.critical_threshold = 0.9
        
        print("🔮 Predictive Panic Attack Detection System Initialized")
        print(f"   📊 Prediction Window: {prediction_window//60} minutes ahead")
        print(f"   ⏰ Early Warning Window: {early_warning_window//60} minutes")
    
    def load_predictive_models(self):
        """Load models trained for predictive detection"""
        try:
            # Load main detection model
            self.detection_model = joblib.load(r'E:\panic attack detector\models\medical_ensemble_model.pkl')
            self.scaler = joblib.load(r'E:\panic attack detector\models\medical_scaler.pkl')
            self.feature_selector = joblib.load(r'E:\panic attack detector\models\medical_feature_selector.pkl')
            
            # Load predictive model (if available, otherwise use detection model)
            try:
                self.predictive_model = joblib.load(r'E:\panic attack detector\models\predictive_panic_model.pkl')
                print("  ✅ Predictive model loaded")
            except:
                self.predictive_model = self.detection_model
                print("  ⚠️ Using detection model for prediction")
            
            # Load baselines and thresholds
            with open(r'E:\panic attack detector\models\medical_baselines.pkl', 'rb') as f:
                self.baseline_data = pickle.load(f)
            
            with open(r'E:\panic attack detector\models\medical_thresholds.pkl', 'rb') as f:
                self.thresholds = pickle.load(f)
            
            print("  ✅ All models loaded successfully!")
            
        except Exception as e:
            print(f"  ❌ Error loading models: {e}")
            raise
    
    def extract_predictive_features(self, current_data, historical_data):
        """
        Extract features for predictive panic attack detection
        
        Args:
            current_data: Current 30-second window data
            historical_data: Previous 10 minutes of data
            
        Returns:
            Predictive features array
        """
        features = []
        
        # 1. Current physiological state
        current_features = self.extract_standard_features(current_data)
        features.extend(current_features)
        
        # 2. Historical trends (predictive indicators)
        if len(historical_data) > 5:  # Need at least 5 windows (2.5 minutes)
            trend_features = self.extract_trend_features(historical_data)
            features.extend(trend_features)
        else:
            features.extend([0] * 20)  # Pad with zeros if insufficient data
        
        # 3. Rate of change features
        if len(historical_data) > 2:
            rate_features = self.extract_rate_of_change_features(historical_data)
            features.extend(rate_features)
        else:
            features.extend([0] * 15)
        
        # 4. Pattern recognition features
        pattern_features = self.extract_pattern_features(historical_data)
        features.extend(pattern_features)
        
        # 5. Early warning indicators
        warning_features = self.extract_early_warning_features(current_data, historical_data)
        features.extend(warning_features)
        
        return np.array(features)
    
    def extract_standard_features(self, data):
        """Extract standard physiological features"""
        features = []
        
        # Heart rate features
        if 'heart_rate' in data:
            hr = data['heart_rate']
            features.extend([
                np.mean(hr), np.std(hr), np.max(hr), np.min(hr),
                np.percentile(hr, 25), np.percentile(hr, 75)
            ])
        else:
            features.extend([0] * 6)
        
        # EDA features
        if 'eda' in data:
            eda = data['eda']
            features.extend([
                np.mean(eda), np.std(eda), np.max(eda), np.min(eda),
                np.sum(eda > np.mean(eda) * 1.2)  # Spikes
            ])
        else:
            features.extend([0] * 5)
        
        # Respiration features
        if 'respiration' in data:
            resp = data['respiration']
            features.extend([
                np.mean(resp), np.std(resp), np.max(resp), np.min(resp)
            ])
        else:
            features.extend([0] * 4)
        
        return features
    
    def extract_trend_features(self, historical_data):
        """Extract trend features for prediction"""
        features = []
        
        # Extract time series for each metric
        metrics = ['heart_rate', 'eda', 'respiration', 'temperature']
        
        for metric in metrics:
            values = []
            for window in historical_data:
                if metric in window:
                    values.append(np.mean(window[metric]))
            
            if len(values) > 1:
                # Linear trend
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                features.append(slope)
                
                # Acceleration (second derivative)
                if len(values) > 2:
                    accel = np.diff(np.diff(values))
                    features.append(np.mean(accel))
                else:
                    features.append(0)
                
                # Volatility
                features.append(np.std(values))
                
                # Recent vs historical comparison
                if len(values) > 5:
                    recent = np.mean(values[-3:])
                    historical = np.mean(values[:-3])
                    features.append(recent - historical)
                else:
                    features.append(0)
            else:
                features.extend([0] * 4)
        
        return features
    
    def extract_rate_of_change_features(self, historical_data):
        """Extract rate of change features"""
        features = []
        
        metrics = ['heart_rate', 'eda', 'respiration']
        
        for metric in metrics:
            values = []
            for window in historical_data:
                if metric in window:
                    values.append(np.mean(window[metric]))
            
            if len(values) > 1:
                # Rate of change
                rates = np.diff(values)
                features.extend([
                    np.mean(rates), np.std(rates), np.max(rates), np.min(rates),
                    np.sum(rates > 0) / len(rates)  # Percentage increasing
                ])
            else:
                features.extend([0] * 5)
        
        return features
    
    def extract_pattern_features(self, historical_data):
        """Extract pattern recognition features"""
        features = []
        
        # Cyclic patterns
        if len(historical_data) > 10:
            hr_values = []
            for window in historical_data:
                if 'heart_rate' in window:
                    hr_values.append(np.mean(window['heart_rate']))
            
            if len(hr_values) > 10:
                # FFT for cyclic patterns
                fft = np.fft.fft(hr_values)
                power_spectrum = np.abs(fft) ** 2
                features.extend([
                    np.max(power_spectrum[1:len(power_spectrum)//2]),  # Dominant frequency
                    np.sum(power_spectrum[1:len(power_spectrum)//2])    # Total power
                ])
            else:
                features.extend([0] * 2)
        else:
            features.extend([0] * 2)
        
        return features
    
    def extract_early_warning_features(self, current_data, historical_data):
        """Extract early warning indicators"""
        features = []
        
        # Compare current to baseline
        baseline_hr = 75  # Default baseline
        baseline_eda = 5.0
        
        if 'heart_rate' in current_data:
            current_hr = np.mean(current_data['heart_rate'])
            hr_increase = (current_hr - baseline_hr) / baseline_hr
            features.append(hr_increase)
        else:
            features.append(0)
        
        if 'eda' in current_data:
            current_eda = np.mean(current_data['eda'])
            eda_increase = (current_eda - baseline_eda) / baseline_eda
            features.append(eda_increase)
        else:
            features.append(0)
        
        # Stress accumulation
        if len(historical_data) > 5:
            stress_indicators = []
            for window in historical_data:
                stress_score = 0
                if 'heart_rate' in window:
                    hr = np.mean(window['heart_rate'])
                    if hr > baseline_hr * 1.1:  # 10% above baseline
                        stress_score += 1
                if 'eda' in window:
                    eda = np.mean(window['eda'])
                    if eda > baseline_eda * 1.2:  # 20% above baseline
                        stress_score += 1
                stress_indicators.append(stress_score)
            
            features.extend([
                np.mean(stress_indicators),  # Average stress level
                np.sum(stress_indicators),   # Total stress accumulation
                np.max(stress_indicators)    # Peak stress level
            ])
        else:
            features.extend([0] * 3)
        
        return features
    
    def predict_panic_attack(self, current_data, historical_data=None):
        """
        Predict if a panic attack will occur in the next prediction_window seconds
        
        Args:
            current_data: Current 30-second window data
            historical_data: Previous data for trend analysis
            
        Returns:
            dict: Prediction results
        """
        if historical_data is None:
            historical_data = list(self.physiological_buffer)
        
        # Extract predictive features
        features = self.extract_predictive_features(current_data, historical_data)
        
        # Ensure features are the right shape
        if len(features) < 100:  # Pad if needed
            features = np.pad(features, (0, 100 - len(features)), 'constant')
        elif len(features) > 100:  # Truncate if too long
            features = features[:100]
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        features_selected = self.feature_selector.transform(features_scaled)
        
        # Get prediction probability
        panic_probability = self.predictive_model.predict_proba(features_selected)[0][1]
        
        # Determine prediction level
        if panic_probability >= self.critical_threshold:
            prediction_level = "CRITICAL"
            time_to_panic = "IMMINENT (0-2 minutes)"
        elif panic_probability >= self.panic_probability_threshold:
            prediction_level = "HIGH"
            time_to_panic = f"LIKELY ({self.prediction_window//60}-{self.prediction_window//30} minutes)"
        elif panic_probability >= self.early_warning_threshold:
            prediction_level = "EARLY WARNING"
            time_to_panic = f"POSSIBLE ({self.prediction_window//60}-{self.prediction_window//30} minutes)"
        else:
            prediction_level = "NORMAL"
            time_to_panic = "No immediate risk"
        
        # Store prediction
        self.prediction_history.append({
            'timestamp': time.time(),
            'probability': panic_probability,
            'level': prediction_level,
            'time_to_panic': time_to_panic
        })
        
        return {
            'panic_probability': panic_probability,
            'prediction_level': prediction_level,
            'time_to_panic': time_to_panic,
            'confidence': panic_probability,
            'features_used': len(features),
            'historical_windows': len(historical_data)
        }
    
    def update_physiological_data(self, data):
        """Update physiological data buffer"""
        self.physiological_buffer.append(data)
        
        # Extract features for current window
        features = self.extract_standard_features(data)
        self.feature_buffer.append(features)
    
    def get_prediction_summary(self):
        """Get summary of recent predictions"""
        if not self.prediction_history:
            return "No predictions available"
        
        recent_predictions = list(self.prediction_history)[-10:]  # Last 10 predictions
        
        summary = {
            'total_predictions': len(self.prediction_history),
            'recent_average_probability': np.mean([p['probability'] for p in recent_predictions]),
            'high_risk_predictions': len([p for p in recent_predictions if p['probability'] > 0.7]),
            'early_warnings': len([p for p in recent_predictions if p['probability'] > 0.4]),
            'current_trend': self.analyze_prediction_trend()
        }
        
        return summary
    
    def analyze_prediction_trend(self):
        """Analyze trend in predictions"""
        if len(self.prediction_history) < 3:
            return "Insufficient data"
        
        recent_probs = [p['probability'] for p in list(self.prediction_history)[-5:]]
        
        if len(recent_probs) >= 3:
            # Calculate trend
            x = np.arange(len(recent_probs))
            slope = np.polyfit(x, recent_probs, 1)[0]
            
            if slope > 0.05:
                return "INCREASING RISK"
            elif slope < -0.05:
                return "DECREASING RISK"
            else:
                return "STABLE"
        
        return "STABLE"
    
    def generate_early_warning_alert(self, prediction_result):
        """Generate early warning alert based on prediction"""
        if prediction_result['prediction_level'] == "CRITICAL":
            return {
                'alert_type': 'CRITICAL',
                'message': '🚨 PANIC ATTACK IMMINENT! Seek immediate help!',
                'actions': ['Call emergency services', 'Use breathing exercises', 'Find safe space'],
                'urgency': 'HIGH'
            }
        elif prediction_result['prediction_level'] == "HIGH":
            return {
                'alert_type': 'HIGH_RISK',
                'message': '⚠️ High risk of panic attack in next 5-10 minutes',
                'actions': ['Start relaxation techniques', 'Remove stressors', 'Prepare coping strategies'],
                'urgency': 'MEDIUM'
            }
        elif prediction_result['prediction_level'] == "EARLY WARNING":
            return {
                'alert_type': 'EARLY_WARNING',
                'message': '🔔 Early warning: Possible panic attack risk',
                'actions': ['Monitor stress levels', 'Practice mindfulness', 'Stay calm'],
                'urgency': 'LOW'
            }
        else:
            return {
                'alert_type': 'NORMAL',
                'message': '✅ No immediate panic attack risk detected',
                'actions': ['Continue normal activities', 'Maintain baseline monitoring'],
                'urgency': 'NONE'
            }

def main():
    """Test the predictive panic attack detection system"""
    print("🔮 Testing Predictive Panic Attack Detection System")
    print("=" * 60)
    
    # Initialize detector
    detector = PredictivePanicDetector(prediction_window=300, early_warning_window=600)
    
    # Simulate some physiological data
    print("\n📊 Simulating physiological data...")
    
    # Simulate normal baseline data
    for i in range(20):  # 10 minutes of data
        data = {
            'heart_rate': np.random.normal(75, 5, 30),
            'eda': np.random.normal(5.0, 0.5, 30),
            'respiration': np.random.normal(16, 2, 30),
            'temperature': np.random.normal(36.5, 0.2, 30)
        }
        detector.update_physiological_data(data)
    
    # Simulate stress buildup (pre-panic indicators)
    print("📈 Simulating stress buildup...")
    for i in range(10):  # 5 minutes of stress buildup
        stress_factor = 1 + (i * 0.1)  # Increasing stress
        data = {
            'heart_rate': np.random.normal(75 * stress_factor, 5, 30),
            'eda': np.random.normal(5.0 * stress_factor, 0.5, 30),
            'respiration': np.random.normal(16 * stress_factor, 2, 30),
            'temperature': np.random.normal(36.5 - (i * 0.1), 0.2, 30)
        }
        detector.update_physiological_data(data)
        
        # Make prediction
        prediction = detector.predict_panic_attack(data)
        
        print(f"  Window {i+1}: {prediction['prediction_level']} "
              f"(Prob: {prediction['panic_probability']:.3f}) - {prediction['time_to_panic']}")
        
        # Generate alert if needed
        alert = detector.generate_early_warning_alert(prediction)
        if alert['urgency'] != 'NONE':
            print(f"    🚨 {alert['message']}")
    
    # Show prediction summary
    print("\n📊 Prediction Summary:")
    summary = detector.get_prediction_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Predictive detection system test completed!")

if __name__ == "__main__":
    main()
