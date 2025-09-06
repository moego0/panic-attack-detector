
// Include libraries
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <SoftwareSerial.h>

// Pin definitions
#define HEART_RATE_PIN A0
#define EDA_PIN A1
#define TEMP_PIN A2
#define RESP_PIN A3
#define SDA_PIN A4
#define SCL_PIN A5

// Sensor objects
Adafruit_MPU6050 mpu;
OneWire oneWire(TEMP_PIN);
DallasTemperature tempSensor(&oneWire);

// Global variables
unsigned long lastReadTime = 0;
const unsigned long readInterval = 100; // 10 Hz (100ms interval)
int heartRate = 0;
float eda = 0.0;
float temperature = 0.0;
float respiration = 0.0;
float accX = 0.0;
float accY = 0.0;
float accZ = 0.0;

// Heart rate calculation variables
int heartRateBuffer[10];
int heartRateIndex = 0;
unsigned long lastHeartBeat = 0;
int heartRateCount = 0;

// EDA calculation variables
float edaBuffer[10];
int edaIndex = 0;

// Respiration calculation variables
float respBuffer[10];
int respIndex = 0;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  while (!Serial) {
    delay(10);
  }
  
  Serial.println("🏥 Real-Time Sensor Reader for Panic Attack Detection");
  Serial.println("==================================================");
  
  // Initialize I2C
  Wire.begin();
  
  // Initialize MPU6050 (accelerometer)
  if (!mpu.begin()) {
    Serial.println("❌ Failed to initialize MPU6050!");
    while (1) {
      delay(10);
    }
  }
  Serial.println("✅ MPU6050 initialized");
  
  // Initialize temperature sensor
  tempSensor.begin();
  Serial.println("✅ Temperature sensor initialized");
  
  // Initialize analog pins
  pinMode(HEART_RATE_PIN, INPUT);
  pinMode(EDA_PIN, INPUT);
  pinMode(RESP_PIN, INPUT);
  
  // Initialize sensor buffers
  for (int i = 0; i < 10; i++) {
    heartRateBuffer[i] = 0;
    edaBuffer[i] = 0.0;
    respBuffer[i] = 0.0;
  }
  
  Serial.println("✅ All sensors initialized");
  Serial.println("📊 Starting data collection...");
  Serial.println("📋 Data format: HR,EDA,RESP,TEMP,ACC_X,ACC_Y,ACC_Z");
  Serial.println("=" * 50);
}

void loop() {
  // Check if it's time to read sensors
  if (millis() - lastReadTime >= readInterval) {
    lastReadTime = millis();
    
    // Read all sensors
    readHeartRate();
    readEDA();
    readTemperature();
    readRespiration();
    readAccelerometer();
    
    // Send data via serial
    sendSensorData();
  }
  
  // Small delay to prevent overwhelming the system
  delay(10);
}

void readHeartRate() {
  // Read raw heart rate sensor value
  int rawValue = analogRead(HEART_RATE_PIN);
  
  // Convert to voltage (0-5V range)
  float voltage = (rawValue * 5.0) / 1024.0;
  
  // Simple heart rate calculation (this is a simplified version)
  // In a real implementation, you would use proper heart rate detection algorithms
  int currentHR = map(rawValue, 0, 1024, 40, 180);
  
  // Add to buffer for smoothing
  heartRateBuffer[heartRateIndex] = currentHR;
  heartRateIndex = (heartRateIndex + 1) % 10;
  
  // Calculate average heart rate
  int sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += heartRateBuffer[i];
  }
  heartRate = sum / 10;
  
  // Detect heart beat (simplified)
  if (voltage > 2.5 && millis() - lastHeartBeat > 300) {
    heartRateCount++;
    lastHeartBeat = millis();
  }
}

void readEDA() {
  // Read EDA sensor value
  int rawValue = analogRead(EDA_PIN);
  
  // Convert to voltage (0-5V range)
  float voltage = (rawValue * 5.0) / 1024.0;
  
  // Convert to microsiemens (simplified conversion)
  float currentEDA = voltage * 10.0; // This is a simplified conversion
  
  // Add to buffer for smoothing
  edaBuffer[edaIndex] = currentEDA;
  edaIndex = (edaIndex + 1) % 10;
  
  // Calculate average EDA
  float sum = 0.0;
  for (int i = 0; i < 10; i++) {
    sum += edaBuffer[i];
  }
  eda = sum / 10.0;
}

void readTemperature() {
  // Request temperature reading
  tempSensor.requestTemperatures();
  
  // Read temperature in Celsius
  temperature = tempSensor.getTempCByIndex(0);
  
  // Check if reading is valid
  if (temperature == DEVICE_DISCONNECTED_C) {
    temperature = 36.5; // Default body temperature
  }
}

void readRespiration() {
  // Read respiration sensor (pressure sensor)
  int rawValue = analogRead(RESP_PIN);
  
  // Convert to voltage (0-5V range)
  float voltage = (rawValue * 5.0) / 1024.0;
  
  // Convert to breathing rate (simplified)
  // In a real implementation, you would use proper breathing rate detection
  float currentResp = map(voltage, 0, 5, 8, 30);
  
  // Add to buffer for smoothing
  respBuffer[respIndex] = currentResp;
  respIndex = (respIndex + 1) % 10;
  
  // Calculate average respiration rate
  float sum = 0.0;
  for (int i = 0; i < 10; i++) {
    sum += respBuffer[i];
  }
  respiration = sum / 10.0;
}

void readAccelerometer() {
  // Read accelerometer data
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  
  // Get acceleration values in m/s²
  accX = a.acceleration.x;
  accY = a.acceleration.y;
  accZ = a.acceleration.z;
}

void sendSensorData() {
  // Send data in CSV format: HR,EDA,RESP,TEMP,ACC_X,ACC_Y,ACC_Z
  Serial.print(heartRate);
  Serial.print(",");
  Serial.print(eda, 2);
  Serial.print(",");
  Serial.print(respiration, 1);
  Serial.print(",");
  Serial.print(temperature, 1);
  Serial.print(",");
  Serial.print(accX, 2);
  Serial.print(",");
  Serial.print(accY, 2);
  Serial.print(",");
  Serial.println(accZ, 2);
}

// Additional functions for sensor calibration and debugging
void calibrateSensors() {
  Serial.println("🔧 Calibrating sensors...");
  
  // Calibrate heart rate sensor
  Serial.println("   ❤️  Calibrating heart rate sensor...");
  delay(2000);
  
  // Calibrate EDA sensor
  Serial.println("   💧 Calibrating EDA sensor...");
  delay(2000);
  
  // Calibrate temperature sensor
  Serial.println("   🌡️  Calibrating temperature sensor...");
  delay(2000);
  
  // Calibrate accelerometer
  Serial.println("   📱 Calibrating accelerometer...");
  delay(2000);
  
  Serial.println("✅ Sensor calibration completed!");
}

void printSensorInfo() {
  Serial.println("📊 Sensor Information:");
  Serial.println("   ❤️  Heart Rate: " + String(heartRate) + " BPM");
  Serial.println("   💧 EDA: " + String(eda, 2) + " μS");
  Serial.println("   🫁 Respiration: " + String(respiration, 1) + " BPM");
  Serial.println("   🌡️  Temperature: " + String(temperature, 1) + "°C");
  Serial.println("   📱 Acceleration: X=" + String(accX, 2) + 
                 " Y=" + String(accY, 2) + " Z=" + String(accZ, 2));
}

// Function to handle serial commands
void handleSerialCommands() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "CALIBRATE") {
      calibrateSensors();
    } else if (command == "INFO") {
      printSensorInfo();
    } else if (command == "START") {
      Serial.println("✅ Data collection started");
    } else if (command == "STOP") {
      Serial.println("⏹️ Data collection stopped");
    } else if (command == "HELP") {
      Serial.println("📋 Available commands:");
      Serial.println("   CALIBRATE - Calibrate all sensors");
      Serial.println("   INFO - Print sensor information");
      Serial.println("   START - Start data collection");
      Serial.println("   STOP - Stop data collection");
      Serial.println("   HELP - Show this help message");
    } else {
      Serial.println("❓ Unknown command: " + command);
      Serial.println("💡 Type HELP for available commands");
    }
  }
}
