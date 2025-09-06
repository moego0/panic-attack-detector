import numpy as np
import pandas as pd
import joblib
import pickle
import os
import time
from datetime import datetime, timedelta
from collections import deque
import json

class UniversalPanicDetector:
    """
    Universal Panic Attack Detection System
    - Anyone can use it
    - Automatic baseline creation for new users
    - Personalized detection for each user
    - Easy setup and usage
    """
    
    def __init__(self, user_id="default_user"):
        """
        Initialize universal detector for any user
        
        Args:
            user_id: Unique identifier for the user (can be name, email, etc.)
        """
        self.user_id = user_id
        self.baseline_data = {}
        self.user_profiles = {}
        self.baseline_window = 1800  # 30 minutes for baseline creation
        self.baseline_samples = deque(maxlen=self.baseline_window)
        
        # Load global models
        self.load_global_models()
        
        # Load or create user profile
        self.load_user_profile()
        
        print(f"🌍 Universal Panic Detection System")
        print(f"👤 User: {self.user_id}")
        print(f"📊 Baseline Status: {'✅ Ready' if self.baseline_data else '⚠️ Needs Calibration'}")
    
    def load_global_models(self):
        """Load the global ML models"""
        try:
            # Load ensemble model
            self.ensemble_model = joblib.load(r'E:\panic attack detector\models\medical_ensemble_model.pkl')
            self.scaler = joblib.load(r'E:\panic attack detector\models\medical_scaler.pkl')
            self.feature_selector = joblib.load(r'E:\panic attack detector\models\medical_feature_selector.pkl')
            
            # Load global thresholds
            with open(r'E:\panic attack detector\models\medical_thresholds.pkl', 'rb') as f:
                self.global_thresholds = pickle.load(f)
            
            print("  ✅ Global models loaded successfully!")
            
        except Exception as e:
            print(f"  ❌ Error loading models: {e}")
            raise
    
    def load_user_profile(self):
        """Load or create user profile"""
        profile_file = f"user_profiles/{self.user_id}_profile.json"
        
        if os.path.exists(profile_file):
            try:
                with open(profile_file, 'r') as f:
                    profile = json.load(f)
                self.user_profiles[self.user_id] = profile
                self.baseline_data = profile.get('baseline', {})
                print(f"  ✅ User profile loaded for {self.user_id}")
            except:
                print(f"  ⚠️ Error loading profile, creating new one")
                self.create_new_user_profile()
        else:
            self.create_new_user_profile()
    
    def create_new_user_profile(self):
        """Create a new user profile"""
        profile = {
            'user_id': self.user_id,
            'created_at': datetime.now().isoformat(),
            'baseline': {},
            'calibration_status': 'pending',
            'total_sessions': 0,
            'panic_detections': 0,
            'last_updated': datetime.now().isoformat()
        }
        
        self.user_profiles[self.user_id] = profile
        self.save_user_profile()
        print(f"  📝 New user profile created for {self.user_id}")
    
    def save_user_profile(self):
        """Save user profile to file"""
        os.makedirs("user_profiles", exist_ok=True)
        profile_file = f"user_profiles/{self.user_id}_profile.json"
        
        with open(profile_file, 'w') as f:
            json.dump(self.user_profiles[self.user_id], f, indent=2)
    
    def start_baseline_calibration(self):
        """
        Start baseline calibration for new user
        User needs to be in a calm, relaxed state for 30 minutes
        """
        print(f"\n🧘 Starting Baseline Calibration for {self.user_id}")
        print("=" * 50)
        print("📋 Instructions for Baseline Calibration:")
        print("1. Find a quiet, comfortable place")
        print("2. Sit or lie down in a relaxed position")
        print("3. Breathe normally and stay calm")
        print("4. Avoid stress, caffeine, or exercise")
        print("5. The system will collect data for 30 minutes")
        print("6. This creates YOUR personal baseline")
        print("\n⏰ Starting 30-minute calibration...")
        
        self.baseline_samples.clear()
        self.calibration_start_time = time.time()
        self.calibration_active = True
        
        return True
    
    def add_calibration_data(self, sensor_data):
        """
        Add sensor data during calibration
        
        Args:
            sensor_data: Dictionary with sensor readings
        """
        if not hasattr(self, 'calibration_active') or not self.calibration_active:
            return False
        
        # Add timestamp
        sensor_data['timestamp'] = time.time()
        self.baseline_samples.append(sensor_data)
        
        # Check if calibration is complete
        elapsed_time = time.time() - self.calibration_start_time
        remaining_time = self.baseline_window - elapsed_time
        
        if remaining_time > 0:
            print(f"  📊 Calibration: {elapsed_time/60:.1f}/30 minutes "
                  f"({len(self.baseline_samples)} samples)")
        else:
            self.complete_baseline_calibration()
        
        return True
    
    def complete_baseline_calibration(self):
        """Complete baseline calibration and calculate personal baseline"""
        if len(self.baseline_samples) < 100:  # Need at least 100 samples
            print("  ❌ Insufficient data for baseline calibration")
            print("  🔄 Please restart calibration")
            return False
        
        print(f"\n📊 Calculating Personal Baseline for {self.user_id}...")
        
        # Extract all sensor data
        heart_rates = []
        eda_values = []
        respiration_rates = []
        temperatures = []
        tremor_values = []
        
        for sample in self.baseline_samples:
            if 'heart_rate' in sample:
                heart_rates.extend(sample['heart_rate'])
            if 'eda' in sample:
                eda_values.extend(sample['eda'])
            if 'respiration' in sample:
                respiration_rates.extend(sample['respiration'])
            if 'temperature' in sample:
                temperatures.extend(sample['temperature'])
            if 'tremor' in sample:
                tremor_values.extend(sample['tremor'])
        
        # Calculate personal baseline
        personal_baseline = {
            'user_id': self.user_id,
            'heart_rate': {
                'mean': np.mean(heart_rates) if heart_rates else 75.0,
                'std': np.std(heart_rates) if heart_rates else 5.0,
                'min': np.min(heart_rates) if heart_rates else 70.0,
                'max': np.max(heart_rates) if heart_rates else 80.0
            },
            'eda': {
                'mean': np.mean(eda_values) if eda_values else 5.0,
                'std': np.std(eda_values) if eda_values else 1.0,
                'min': np.min(eda_values) if eda_values else 3.0,
                'max': np.max(eda_values) if eda_values else 7.0
            },
            'respiration': {
                'mean': np.mean(respiration_rates) if respiration_rates else 16.0,
                'std': np.std(respiration_rates) if respiration_rates else 2.0,
                'min': np.min(respiration_rates) if respiration_rates else 12.0,
                'max': np.max(respiration_rates) if respiration_rates else 20.0
            },
            'temperature': {
                'mean': np.mean(temperatures) if temperatures else 36.5,
                'std': np.std(temperatures) if temperatures else 0.3,
                'min': np.min(temperatures) if temperatures else 36.0,
                'max': np.max(temperatures) if temperatures else 37.0
            },
            'tremor': {
                'mean': np.mean(tremor_values) if tremor_values else 0.0,
                'std': np.std(tremor_values) if tremor_values else 0.1,
                'variance': np.var(tremor_values) if tremor_values else 0.0
            },
            'calibration_date': datetime.now().isoformat(),
            'samples_used': len(self.baseline_samples)
        }
        
        # Save baseline
        self.baseline_data = personal_baseline
        self.user_profiles[self.user_id]['baseline'] = personal_baseline
        self.user_profiles[self.user_id]['calibration_status'] = 'completed'
        self.user_profiles[self.user_id]['last_updated'] = datetime.now().isoformat()
        
        self.save_user_profile()
        self.calibration_active = False
        
        print("  ✅ Personal baseline created successfully!")
        print(f"  ❤️  Your Heart Rate: {personal_baseline['heart_rate']['mean']:.1f} ± {personal_baseline['heart_rate']['std']:.1f} BPM")
        print(f"  💧 Your EDA: {personal_baseline['eda']['mean']:.3f} ± {personal_baseline['eda']['std']:.3f} μS")
        print(f"  🫁 Your Breathing: {personal_baseline['respiration']['mean']:.1f} ± {personal_baseline['respiration']['std']:.1f} BPM")
        print(f"  🌡️  Your Temperature: {personal_baseline['temperature']['mean']:.1f}°C")
        
        return True
    
    def detect_panic_attack(self, sensor_data):
        """
        Detect panic attack using personal baseline
        
        Args:
            sensor_data: Current sensor readings
            
        Returns:
            Detection results with personal thresholds
        """
        if not self.baseline_data:
            return {
                'panic_detected': False,
                'confidence': 0.0,
                'message': '⚠️ No baseline available. Please complete calibration first.',
                'personal_thresholds': {},
                'current_values': {},
                'deviations': {}
            }
        
        # Extract current values
        current_values = self.extract_current_values(sensor_data)
        
        # Calculate deviations from personal baseline
        deviations = self.calculate_deviations(current_values)
        
        # Check personal panic thresholds
        panic_indicators = self.check_personal_thresholds(deviations)
        
        # Calculate overall panic probability
        panic_probability = self.calculate_panic_probability(panic_indicators, deviations)
        
        # Determine if panic is detected
        panic_detected = panic_probability > 0.7
        
        # Generate personalized message
        message = self.generate_personalized_message(panic_detected, panic_probability, deviations)
        
        return {
            'panic_detected': panic_detected,
            'confidence': panic_probability,
            'message': message,
            'personal_thresholds': self.get_personal_thresholds(),
            'current_values': current_values,
            'deviations': deviations,
            'panic_indicators': panic_indicators
        }
    
    def extract_current_values(self, sensor_data):
        """Extract current sensor values"""
        return {
            'heart_rate': np.mean(sensor_data.get('heart_rate', [75])) if sensor_data.get('heart_rate') else 75,
            'eda': np.mean(sensor_data.get('eda', [5.0])) if sensor_data.get('eda') else 5.0,
            'respiration': np.mean(sensor_data.get('respiration', [16])) if sensor_data.get('respiration') else 16,
            'temperature': np.mean(sensor_data.get('temperature', [36.5])) if sensor_data.get('temperature') else 36.5,
            'tremor': np.var(sensor_data.get('tremor', [0])) if sensor_data.get('tremor') else 0
        }
    
    def calculate_deviations(self, current_values):
        """Calculate deviations from personal baseline"""
        baseline = self.baseline_data
        
        deviations = {}
        for metric in ['heart_rate', 'eda', 'respiration', 'temperature', 'tremor']:
            if metric in current_values and metric in baseline:
                baseline_mean = baseline[metric]['mean']
                current_value = current_values[metric]
                
                if baseline_mean > 0:
                    deviation = (current_value - baseline_mean) / baseline_mean
                else:
                    deviation = 0
                
                deviations[metric] = {
                    'absolute': current_value - baseline_mean,
                    'percentage': deviation * 100,
                    'z_score': (current_value - baseline_mean) / baseline[metric]['std'] if baseline[metric]['std'] > 0 else 0
                }
        
        return deviations
    
    def check_personal_thresholds(self, deviations):
        """Check against personal panic thresholds"""
        indicators = {}
        
        # Heart rate threshold (25% above personal baseline)
        hr_deviation = deviations.get('heart_rate', {}).get('percentage', 0)
        indicators['heart_rate'] = hr_deviation > 25
        
        # EDA threshold (30% above personal baseline)
        eda_deviation = deviations.get('eda', {}).get('percentage', 0)
        indicators['eda'] = eda_deviation > 30
        
        # Respiration threshold (20% above personal baseline)
        resp_deviation = deviations.get('respiration', {}).get('percentage', 0)
        indicators['respiration'] = resp_deviation > 20
        
        # Temperature threshold (significant drop)
        temp_deviation = deviations.get('temperature', {}).get('percentage', 0)
        indicators['temperature'] = temp_deviation < -5  # 5% drop
        
        # Tremor threshold (significant increase)
        tremor_deviation = deviations.get('tremor', {}).get('percentage', 0)
        indicators['tremor'] = tremor_deviation > 50  # 50% increase
        
        return indicators
    
    def calculate_panic_probability(self, indicators, deviations):
        """Calculate overall panic probability"""
        # Count positive indicators
        positive_indicators = sum(indicators.values())
        total_indicators = len(indicators)
        
        # Base probability from indicator count
        base_probability = positive_indicators / total_indicators
        
        # Adjust based on deviation severity
        max_deviation = max([abs(d.get('percentage', 0)) for d in deviations.values()])
        severity_factor = min(max_deviation / 100, 1.0)  # Cap at 1.0
        
        # Combine base probability with severity
        panic_probability = base_probability * 0.7 + severity_factor * 0.3
        
        return min(panic_probability, 1.0)
    
    def get_personal_thresholds(self):
        """Get personal panic thresholds"""
        if not self.baseline_data:
            return {}
        
        baseline = self.baseline_data
        return {
            'heart_rate': f"{baseline['heart_rate']['mean'] * 1.25:.1f} BPM (25% above your {baseline['heart_rate']['mean']:.1f} BPM)",
            'eda': f"{baseline['eda']['mean'] * 1.30:.3f} μS (30% above your {baseline['eda']['mean']:.3f} μS)",
            'respiration': f"{baseline['respiration']['mean'] * 1.20:.1f} BPM (20% above your {baseline['respiration']['mean']:.1f} BPM)",
            'temperature': f"{baseline['temperature']['mean'] * 0.95:.1f}°C (5% below your {baseline['temperature']['mean']:.1f}°C)",
            'tremor': f"{baseline['tremor']['variance'] * 1.50:.3f} (50% above your {baseline['tremor']['variance']:.3f})"
        }
    
    def generate_personalized_message(self, panic_detected, confidence, deviations):
        """Generate personalized message based on detection"""
        if panic_detected:
            return f"🚨 PANIC ATTACK DETECTED! (Confidence: {confidence:.1%}) - Seek help immediately!"
        elif confidence > 0.5:
            return f"⚠️ High stress detected (Confidence: {confidence:.1%}) - Consider relaxation techniques"
        elif confidence > 0.3:
            return f"🔔 Elevated stress levels (Confidence: {confidence:.1%}) - Monitor your condition"
        else:
            return f"✅ Normal stress levels (Confidence: {confidence:.1%}) - You're doing well!"
    
    def get_user_status(self):
        """Get current user status"""
        if not self.baseline_data:
            return {
                'status': 'needs_calibration',
                'message': 'Please complete 30-minute baseline calibration',
                'baseline_ready': False
            }
        else:
            return {
                'status': 'ready',
                'message': 'System ready for panic detection',
                'baseline_ready': True,
                'baseline_info': {
                    'heart_rate': f"{self.baseline_data['heart_rate']['mean']:.1f} BPM",
                    'eda': f"{self.baseline_data['eda']['mean']:.3f} μS",
                    'respiration': f"{self.baseline_data['respiration']['mean']:.1f} BPM",
                    'calibration_date': self.baseline_data['calibration_date']
                }
            }

def main():
    """Demo of Universal Panic Detection System"""
    print("🌍 Universal Panic Attack Detection System Demo")
    print("=" * 60)
    
    # Create detector for a new user
    user_id = "demo_user"
    detector = UniversalPanicDetector(user_id)
    
    # Check user status
    status = detector.get_user_status()
    print(f"\n👤 User Status: {status['status']}")
    print(f"📝 Message: {status['message']}")
    
    if not status['baseline_ready']:
        print("\n🧘 Starting baseline calibration...")
        detector.start_baseline_calibration()
        
        # Simulate calibration data
        print("📊 Simulating calibration data...")
        for i in range(30):  # Simulate 30 minutes of data
            sensor_data = {
                'heart_rate': np.random.normal(75, 5, 30),
                'eda': np.random.normal(5.0, 1.0, 30),
                'respiration': np.random.normal(16, 2, 30),
                'temperature': np.random.normal(36.5, 0.3, 30),
                'tremor': np.random.normal(0, 0.1, 30)
            }
            detector.add_calibration_data(sensor_data)
            time.sleep(0.1)  # Simulate time passing
    
    # Test panic detection
    print("\n🔍 Testing panic detection...")
    
    # Normal data
    normal_data = {
        'heart_rate': np.random.normal(75, 5, 30),
        'eda': np.random.normal(5.0, 1.0, 30),
        'respiration': np.random.normal(16, 2, 30),
        'temperature': np.random.normal(36.5, 0.3, 30),
        'tremor': np.random.normal(0, 0.1, 30)
    }
    
    result = detector.detect_panic_attack(normal_data)
    print(f"Normal data: {result['message']}")
    
    # Stress data
    stress_data = {
        'heart_rate': np.random.normal(95, 8, 30),  # Higher HR
        'eda': np.random.normal(8.0, 2.0, 30),     # Higher EDA
        'respiration': np.random.normal(22, 3, 30), # Higher respiration
        'temperature': np.random.normal(36.0, 0.5, 30), # Lower temp
        'tremor': np.random.normal(0.5, 0.3, 30)   # Higher tremor
    }
    
    result = detector.detect_panic_attack(stress_data)
    print(f"Stress data: {result['message']}")
    
    print("\n✅ Universal detection system demo completed!")

if __name__ == "__main__":
    main()
