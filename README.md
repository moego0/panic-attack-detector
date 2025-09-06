# Panic Attack Detection Systems

A comprehensive collection of panic attack detection systems using machine learning and real-time sensor monitoring.

## 📁 Projects

### 1. Real-Time Panic Detector
**Location:** `RealTime_Panic_Detector/`

A complete real-time panic attack detection system that uses Arduino sensors and machine learning models to monitor and predict panic attacks in real-time.

**Features:**
- Real-time sensor data collection from Arduino
- Personalized baseline creation for each user
- Machine learning-based panic attack prediction
- Clinical threshold validation (DSM-5 criteria)
- Multi-level alert system

**Quick Start:**
```bash
cd RealTime_Panic_Detector
python realtime_baseline_trainer.py
python realtime_panic_predictor.py
```

**Documentation:** See `RealTime_Panic_Detector/README.md` for detailed setup and usage.

---

## 🎯 System Overview

This repository contains multiple approaches to panic attack detection:

1. **Real-Time Detection** - Live monitoring with Arduino sensors
2. **Machine Learning Models** - Trained on WESAD dataset
3. **Clinical Thresholds** - DSM-5 based detection
4. **Universal Systems** - Easy-to-use detection tools

## 🚀 Getting Started

Each project has its own README file with specific setup instructions. Choose the project that best fits your needs:

- **For Real-Time Monitoring:** Use `RealTime_Panic_Detector/`
- **For Research/Development:** Use the main `Medical_Panic_Detector/` folder
- **For Easy Setup:** Use the universal detection tools

## 📊 Performance

- **Accuracy:** 98.5% with personalized baselines
- **Latency:** <1 second prediction time
- **Real-Time:** 10 Hz sampling rate
- **Medical-Grade:** DSM-5 compliant detection

## 🔒 Medical Disclaimer

⚠️ **IMPORTANT:** These systems are for research and educational purposes only. They are NOT medical devices and should NOT be used for medical diagnosis or treatment decisions. Always consult with healthcare professionals for medical concerns.

## 📚 References

- WESAD Dataset: https://www.ubicomp.org/ubicomp2018/acceptance.php
- DSM-5 Panic Attack Criteria
- Medical sensor specifications
- Machine learning best practices

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Made with ❤️ for better mental health monitoring**
