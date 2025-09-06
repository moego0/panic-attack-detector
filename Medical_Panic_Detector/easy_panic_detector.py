"""
Easy Panic Attack Detection System
- Anyone can use it
- Simple setup
- Automatic baseline creation
- Personal detection for each user
"""

import numpy as np
import time
from universal_panic_detector import UniversalPanicDetector

class EasyPanicDetector:
    """Simple interface for anyone to use panic detection"""
    
    def __init__(self):
        self.detector = None
        self.current_user = None
    
    def setup_new_user(self, user_name):
        """Setup for a new user"""
        print(f"👋 Welcome {user_name}!")
        print("Let's set up your personal panic detection system.")
        
        # Create detector for this user
        self.detector = UniversalPanicDetector(user_name)
        self.current_user = user_name
        
        # Check if they need calibration
        status = self.detector.get_user_status()
        
        if not status['baseline_ready']:
            print(f"\n📋 Setup Required:")
            print(f"   {status['message']}")
            return self.start_calibration()
        else:
            print(f"\n✅ You're all set up!")
            print(f"   Your personal baseline is ready.")
            return True
    
    def start_calibration(self):
        """Start the 30-minute calibration process"""
        print(f"\n🧘 Baseline Calibration for {self.current_user}")
        print("=" * 50)
        print("📋 What you need to do:")
        print("1. Find a quiet, comfortable place")
        print("2. Sit or lie down and relax")
        print("3. Breathe normally")
        print("4. Stay calm for 30 minutes")
        print("5. The system will learn YOUR normal levels")
        print("\n⏰ This creates your personal baseline for accurate detection")
        
        response = input("\n🤔 Are you ready to start? (y/n): ").lower()
        
        if response == 'y':
            print("\n🚀 Starting calibration...")
            self.detector.start_baseline_calibration()
            return self.run_calibration()
        else:
            print("👋 No problem! You can start calibration anytime.")
            return False
    
    def run_calibration(self):
        """Run the calibration process"""
        print("\n📊 Calibration in progress...")
        print("   (In real use, this would connect to your sensors)")
        
        # Simulate calibration data collection
        for minute in range(30):
            print(f"   ⏰ Minute {minute + 1}/30: Collecting your baseline data...")
            
            # Simulate sensor data (in real use, this comes from Arduino)
            sensor_data = {
                'heart_rate': np.random.normal(75, 5, 30),
                'eda': np.random.normal(5.0, 1.0, 30),
                'respiration': np.random.normal(16, 2, 30),
                'temperature': np.random.normal(36.5, 0.3, 30),
                'tremor': np.random.normal(0, 0.1, 30)
            }
            
            self.detector.add_calibration_data(sensor_data)
            time.sleep(0.1)  # Simulate time passing
        
        print("\n✅ Calibration complete!")
        return True
    
    def start_monitoring(self):
        """Start monitoring for panic attacks"""
        if not self.detector:
            print("❌ Please setup a user first!")
            return
        
        status = self.detector.get_user_status()
        if not status['baseline_ready']:
            print("❌ Please complete calibration first!")
            return
        
        print(f"\n🔍 Starting panic attack monitoring for {self.current_user}")
        print("=" * 50)
        print("📊 Your personal thresholds:")
        thresholds = self.detector.get_personal_thresholds()
        for metric, threshold in thresholds.items():
            print(f"   {metric}: {threshold}")
        
        print(f"\n🚨 Monitoring active...")
        print("   (In real use, this would connect to your sensors)")
        print("   Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                # Simulate real-time monitoring
                self.simulate_monitoring()
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print(f"\n👋 Monitoring stopped for {self.current_user}")
    
    def simulate_monitoring(self):
        """Simulate real-time monitoring"""
        # Simulate different scenarios
        scenarios = [
            # Normal scenario
            {
                'name': 'Normal',
                'data': {
                    'heart_rate': np.random.normal(75, 5, 30),
                    'eda': np.random.normal(5.0, 1.0, 30),
                    'respiration': np.random.normal(16, 2, 30),
                    'temperature': np.random.normal(36.5, 0.3, 30),
                    'tremor': np.random.normal(0, 0.1, 30)
                }
            },
            # Stress scenario
            {
                'name': 'Stress',
                'data': {
                    'heart_rate': np.random.normal(90, 8, 30),
                    'eda': np.random.normal(7.0, 2.0, 30),
                    'respiration': np.random.normal(20, 3, 30),
                    'temperature': np.random.normal(36.2, 0.5, 30),
                    'tremor': np.random.normal(0.3, 0.2, 30)
                }
            },
            # Panic scenario
            {
                'name': 'Panic',
                'data': {
                    'heart_rate': np.random.normal(110, 10, 30),
                    'eda': np.random.normal(12.0, 3.0, 30),
                    'respiration': np.random.normal(28, 4, 30),
                    'temperature': np.random.normal(35.8, 0.8, 30),
                    'tremor': np.random.normal(1.0, 0.5, 30)
                }
            }
        ]
        
        # Randomly select a scenario
        scenario = np.random.choice(scenarios)
        
        # Get detection result
        result = self.detector.detect_panic_attack(scenario['data'])
        
        # Display result
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {scenario['name']}: {result['message']}")
        
        if result['panic_detected']:
            print(f"   🚨 PANIC DETECTED! Confidence: {result['confidence']:.1%}")
            print(f"   📊 Deviations: {result['deviations']}")
    
    def show_user_info(self):
        """Show current user information"""
        if not self.detector:
            print("❌ No user setup!")
            return
        
        status = self.detector.get_user_status()
        print(f"\n👤 User: {self.current_user}")
        print(f"📊 Status: {status['status']}")
        print(f"📝 Message: {status['message']}")
        
        if status['baseline_ready']:
            print(f"\n📈 Your Personal Baseline:")
            for metric, value in status['baseline_info'].items():
                print(f"   {metric}: {value}")

def main():
    """Main interface for easy panic detection"""
    print("🌍 Easy Panic Attack Detection System")
    print("=" * 50)
    print("👋 Welcome! This system can detect panic attacks for anyone.")
    print("📊 Each person gets their own personalized baseline.")
    print("🔍 The system learns YOUR normal levels for accurate detection.")
    
    detector = EasyPanicDetector()
    
    while True:
        print(f"\n📋 What would you like to do?")
        print("1. Setup new user")
        print("2. Start monitoring")
        print("3. Show user info")
        print("4. Exit")
        
        choice = input("\n🤔 Enter your choice (1-4): ").strip()
        
        if choice == '1':
            user_name = input("👤 Enter your name: ").strip()
            if user_name:
                detector.setup_new_user(user_name)
            else:
                print("❌ Please enter a valid name!")
        
        elif choice == '2':
            detector.start_monitoring()
        
        elif choice == '3':
            detector.show_user_info()
        
        elif choice == '4':
            print("👋 Thank you for using the Panic Detection System!")
            break
        
        else:
            print("❌ Invalid choice! Please enter 1-4.")

if __name__ == "__main__":
    main()
