import pickle
import numpy as np

# Load baselines
with open(r'E:\panic attack detector\models\medical_baselines.pkl', 'rb') as f:
    baselines = pickle.load(f)

print("📊 Individual Baselines for Each Person:")
print("=" * 50)

for subject_id, baseline in baselines.items():
    print(f"\n👤 Subject {subject_id}:")
    print(f"  ❤️  Heart Rate: {baseline['heart_rate']:.1f} BPM")
    print(f"  💧 EDA: {baseline['eda']['mean']:.3f} μS")
    print(f"  🫁 Breathing Rate: {baseline['breathing_rate']:.1f} BPM")
    print(f"  🌡️  Temperature: {baseline['skin_temp']['mean']:.1f}°C")
    print(f"  🤲 Tremor: {baseline['tremor']['variance']:.3f}")

print(f"\n✅ Total subjects with baselines: {len(baselines)}")
print("📈 Each person has their own personalized baseline!")
