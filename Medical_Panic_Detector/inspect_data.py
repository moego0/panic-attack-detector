import pickle
import numpy as np

# Load one subject to inspect structure
try:
    with open('WESAD/S2/S2.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    print("🔍 WESAD Data Structure Inspection")
    print("=" * 50)
    print(f"Keys: {list(data.keys())}")
    print("\nData details:")
    
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        elif hasattr(value, '__len__'):
            print(f"  {key}: length {len(value)}, type {type(value)}")
        else:
            print(f"  {key}: {type(value)} = {value}")
    
    # Check labels specifically
    if 'label' in data:
        labels = data['label']
        print(f"\n📊 Label Analysis:")
        print(f"  Unique labels: {np.unique(labels)}")
        print(f"  Label counts: {np.bincount(labels)}")
        
except Exception as e:
    print(f"❌ Error loading data: {e}")
