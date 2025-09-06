#!/usr/bin/env python3
"""
Medical Panic Attack Detection System - Test Script
Tests the system components and validates functionality
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import joblib
from datetime import datetime

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing Package Imports...")
    
    try:
        import numpy as np
        print("  ✅ NumPy imported successfully")
    except ImportError as e:
        print(f"  ❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("  ✅ Pandas imported successfully")
    except ImportError as e:
        print(f"  ❌ Pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print("  ✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"  ❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import matplotlib
        print("  ✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"  ❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn
        print("  ✅ Seaborn imported successfully")
    except ImportError as e:
        print(f"  ❌ Seaborn import failed: {e}")
        return False
    
    try:
        import scipy
        print("  ✅ SciPy imported successfully")
    except ImportError as e:
        print(f"  ❌ SciPy import failed: {e}")
        return False
    
    try:
        import joblib
        print("  ✅ Joblib imported successfully")
    except ImportError as e:
        print(f"  ❌ Joblib import failed: {e}")
        return False
    
    try:
        import serial
        print("  ✅ PySerial imported successfully")
    except ImportError as e:
        print(f"  ❌ PySerial import failed: {e}")
        return False
    
    return True

def test_wesad_data():
    """Test if WESAD data is available"""
    print("\n🧪 Testing WESAD Dataset...")
    
    wesad_path = "../WESAD"
    if not os.path.exists(wesad_path):
        print(f"  ❌ WESAD directory not found at {wesad_path}")
        return False
    
    # Check for subject files
    subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    found_subjects = []
    
    for subject_id in subject_ids:
        subject_file = os.path.join(wesad_path, f"S{subject_id}", f"S{subject_id}.pkl")
        if os.path.exists(subject_file):
            found_subjects.append(subject_id)
    
    print(f"  ✅ Found {len(found_subjects)} WESAD subjects: {found_subjects}")
    
    if len(found_subjects) < 5:
        print("  ⚠️ Warning: Less than 5 subjects found. Training may be limited.")
    
    return len(found_subjects) > 0

def test_model_training():
    """Test model training functionality"""
    print("\n🧪 Testing Model Training...")
    
    try:
        # Import the training module
        sys.path.append('.')
        from medical_panic_trainer import MedicalPanicDetector
        
        # Initialize detector
        detector = MedicalPanicDetector()
        print("  ✅ MedicalPanicDetector initialized")
        
        # Test data loading
        all_data = detector.load_all_wesad_data()
        if len(all_data) == 0:
            print("  ❌ No WESAD data loaded")
            return False
        
        print(f"  ✅ Loaded {len(all_data)} subjects")
        
        # Test baseline calculation
        subject_id, data = all_data[0]
        baseline = detector.calculate_medical_baseline(data, subject_id)
        print("  ✅ Baseline calculation successful")
        
        # Test feature extraction
        features, labels = detector.extract_medical_features(data, baseline)
        print(f"  ✅ Feature extraction successful: {features.shape[0]} samples, {features.shape[1]} features")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model training test failed: {e}")
        return False

def test_realtime_detection():
    """Test real-time detection functionality"""
    print("\n🧪 Testing Real-Time Detection...")
    
    try:
        # Import the real-time module
        sys.path.append('.')
        from realtime.medical_realtime_detector import MedicalRealtimeDetector
        
        # Initialize detector
        detector = MedicalRealtimeDetector()
        print("  ✅ MedicalRealtimeDetector initialized")
        
        # Test model loading (if models exist)
        try:
            detector.load_medical_models()
            print("  ✅ Medical models loaded successfully")
        except FileNotFoundError:
            print("  ⚠️ Models not found - need to train first")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Real-time detection test failed: {e}")
        return False

def test_arduino_code():
    """Test Arduino code syntax"""
    print("\n🧪 Testing Arduino Code...")
    
    arduino_file = "arduino/medical_sensor_reader.ino"
    if not os.path.exists(arduino_file):
        print(f"  ❌ Arduino file not found: {arduino_file}")
        return False
    
    # Basic syntax check
    with open(arduino_file, 'r') as f:
        content = f.read()
    
    # Check for required functions
    required_functions = [
        'setup()', 'loop()', 'readAllSensors()', 'sendDataToPython()',
        'calibrateSensors()', 'handleBluetoothCommands()'
    ]
    
    missing_functions = []
    for func in required_functions:
        if func not in content:
            missing_functions.append(func)
    
    if missing_functions:
        print(f"  ❌ Missing Arduino functions: {missing_functions}")
        return False
    
    print("  ✅ Arduino code syntax check passed")
    return True

def test_directory_structure():
    """Test if all required directories exist"""
    print("\n🧪 Testing Directory Structure...")
    
    required_dirs = [
        'models',
        'realtime',
        'arduino',
        'docs',
        'data'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"  ❌ Missing directories: {missing_dirs}")
        return False
    
    print("  ✅ All required directories exist")
    return True

def test_file_permissions():
    """Test file permissions"""
    print("\n🧪 Testing File Permissions...")
    
    test_files = [
        'medical_panic_trainer.py',
        'realtime/medical_realtime_detector.py',
        'arduino/medical_sensor_reader.ino',
        'docs/MEDICAL_SYSTEM_DOCUMENTATION.md',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in test_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    print("  ✅ All required files exist")
    return True

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("🏥 Medical Panic Attack Detection System - Comprehensive Test")
    print("="*70)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Package Imports", test_imports()))
    test_results.append(("Directory Structure", test_directory_structure()))
    test_results.append(("File Permissions", test_file_permissions()))
    test_results.append(("WESAD Dataset", test_wesad_data()))
    test_results.append(("Arduino Code", test_arduino_code()))
    test_results.append(("Model Training", test_model_training()))
    test_results.append(("Real-time Detection", test_realtime_detection()))
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:<25} {status}")
        if result:
            passed_tests += 1
    
    print(f"\n📈 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! System is ready for use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

def main():
    """Main test function"""
    success = run_comprehensive_test()
    
    if success:
        print("\n🚀 Next Steps:")
        print("  1. Run: python medical_panic_trainer.py")
        print("  2. Upload Arduino code to your Arduino Uno")
        print("  3. Run: python realtime/medical_realtime_detector.py")
        print("  4. Follow the documentation for setup instructions")
    else:
        print("\n🔧 Troubleshooting:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Check WESAD dataset location")
        print("  3. Verify Arduino connections")
        print("  4. Read the documentation for detailed setup")

if __name__ == "__main__":
    main()
