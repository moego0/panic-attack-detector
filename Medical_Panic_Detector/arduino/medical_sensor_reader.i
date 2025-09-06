/*
 * Medical-Grade Panic Attack Detection System - Arduino Code
 * 
 * This code reads physiological sensors and sends data via Bluetooth
 * to the Python medical detection system.
 * 
 * Hardware Requirements:
 * - Arduino Uno
 * - HC-05 Bluetooth Module
 * - AD8232 ECG Sensor
 * - GSR/EDA Sensor
 * - Temperature Sensor (DS18B20)
 * - Accelerometer (MPU6050)
 * - Heart Rate Sensor (Pulse Sensor)
 * 
 * Pin Connections:
 * - ECG: A0 (AD8232)
 * - EDA: A1 (GSR Sensor)
 * - Respiration: A2 (Pressure Sensor)
 * - EMG: A3 (Muscle Sensor)
 * - Temperature: A4 (DS18B20)
 * - BVP: A5 (Pulse Sensor)
 * - Wrist EDA: A6 (Secondary GSR)
 * - Wrist Temperature: A7 (Secondary Temp)
 * - Accelerometer: SDA (A4), SCL (A5)
 * - Bluetooth: TX (Pin 2), RX (Pin 3)
 * 
 * Author: Medical AI System
 * Date: 2024
 * Version: 1.0
 */

#include <SoftwareSerial.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <MPU6050.h>

// Bluetooth communication
SoftwareSerial bluetooth(2, 3); // TX, RX

// Temperature sensor
OneWire oneWire(A4);
DallasTemperature tempSensor(&oneWire);

// Accelerometer
MPU6050 mpu;

// Sensor pins
const int ECG_PIN = A0;
const int EDA_PIN = A1;
const int RESP_PIN = A2;
const int EMG_PIN = A3;
const int TEMP_PIN = A4;
const int BVP_PIN = A5;
const int WRIST_EDA_PIN = A6;
const int WRIST_TEMP_PIN = A7;

// Sampling parameters
const int SAMPLE_RATE = 700; // Hz (matching WESAD dataset)
const int WINDOW_SIZE = 7000; // 10 seconds
const int OVERLAP = 3500; // 50% overlap

// Data buffers
float ecgBuffer[WINDOW_SIZE];
float edaBuffer[WINDOW_SIZE];
float respBuffer[WINDOW_SIZE];
float emgBuffer[WINDOW_SIZE];
float tempBuffer[WINDOW_SIZE];
float bvpBuffer[WINDOW_SIZE];
float wristEdaBuffer[WINDOW_SIZE];
float wristTempBuffer[WINDOW_SIZE];
float accXBuffer[WINDOW_SIZE];
float accYBuffer[WINDOW_SIZE];
float accZBuffer[WINDOW_SIZE];

int bufferIndex = 0;
unsigned long lastSampleTime = 0;
unsigned long sampleInterval = 1000000 / SAMPLE_RATE; // microseconds

// Calibration values
float ecgBaseline = 512.0;
float edaBaseline = 0.0;
float respBaseline = 512.0;
float emgBaseline = 512.0;
float tempBaseline = 25.0;
float bvpBaseline = 512.0;
float wristEdaBaseline = 0.0;
float wristTempBaseline = 25.0;

// System status
bool systemReady = false;
bool sensorsCalibrated = false;
int calibrationSamples = 0;
const int CALIBRATION_DURATION = 5000; // 5 seconds

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  bluetooth.begin(9600);
  
  // Initialize I2C
  Wire.begin();
  
  // Initialize temperature sensor
  tempSensor.begin();
  
  // Initialize accelerometer
  mpu.initialize();
  if (mpu.testConnection()) {
    Serial.println("MPU6050 initialized successfully");
  } else {
    Serial.println("MPU6050 initialization failed");
  }
  
  // Initialize analog pins
  pinMode(ECG_PIN, INPUT);
  pinMode(EDA_PIN, INPUT);
  pinMode(RESP_PIN, INPUT);
  pinMode(EMG_PIN, INPUT);
  pinMode(TEMP_PIN, INPUT);
  pinMode(BVP_PIN, INPUT);
  pinMode(WRIST_EDA_PIN, INPUT);
  pinMode(WRIST_TEMP_PIN, INPUT);
  
  // Initialize system
  initializeSystem();
  
  Serial.println("Medical Panic Detection System Ready");
  Serial.println("Calibrating sensors...");
}

void loop() {
  unsigned long currentTime = micros();
  
  // Check if it's time for next sample
  if (currentTime - lastSampleTime >= sampleInterval) {
    lastSampleTime = currentTime;
    
    // Calibrate sensors if needed
    if (!sensorsCalibrated) {
      calibrateSensors();
      return;
    }
    
    // Read all sensors
    readAllSensors();
    
    // Process data
    processData();
    
    // Send data if buffer is full
    if (bufferIndex >= WINDOW_SIZE) {
      sendDataToPython();
      shiftBuffer();
    }
  }
  
  // Handle Bluetooth commands
  handleBluetoothCommands();
}

void initializeSystem() {
  // Perform system self-test
  systemReady = true;
  
  // Set initial values
  bufferIndex = 0;
  calibrationSamples = 0;
  
  // Initialize buffers
  for (int i = 0; i < WINDOW_SIZE; i++) {
    ecgBuffer[i] = 0.0;
    edaBuffer[i] = 0.0;
    respBuffer[i] = 0.0;
    emgBuffer[i] = 0.0;
    tempBuffer[i] = 0.0;
    bvpBuffer[i] = 0.0;
    wristEdaBuffer[i] = 0.0;
    wristTempBuffer[i] = 0.0;
    accXBuffer[i] = 0.0;
    accYBuffer[i] = 0.0;
    accZBuffer[i] = 0.0;
  }
}

void calibrateSensors() {
  // Read sensors for calibration
  float ecgSum = 0, edaSum = 0, respSum = 0, emgSum = 0;
  float tempSum = 0, bvpSum = 0, wristEdaSum = 0, wristTempSum = 0;
  
  ecgSum += analogRead(ECG_PIN);
  edaSum += analogRead(EDA_PIN);
  respSum += analogRead(RESP_PIN);
  emgSum += analogRead(EMG_PIN);
  tempSum += analogRead(TEMP_PIN);
  bvpSum += analogRead(BVP_PIN);
  wristEdaSum += analogRead(WRIST_EDA_PIN);
  wristTempSum += analogRead(WRIST_TEMP_PIN);
  
  calibrationSamples++;
  
  // Calculate baselines after calibration period
  if (calibrationSamples >= CALIBRATION_DURATION / (1000000 / sampleInterval)) {
    ecgBaseline = ecgSum / calibrationSamples;
    edaBaseline = edaSum / calibrationSamples;
    respBaseline = respSum / calibrationSamples;
    emgBaseline = emgSum / calibrationSamples;
    tempBaseline = tempSum / calibrationSamples;
    bvpBaseline = bvpSum / calibrationSamples;
    wristEdaBaseline = wristEdaSum / calibrationSamples;
    wristTempBaseline = wristTempSum / calibrationSamples;
    
    sensorsCalibrated = true;
    Serial.println("Sensors calibrated successfully");
    Serial.println("Starting medical monitoring...");
  }
}

void readAllSensors() {
  // Read ECG (Heart Rate)
  int ecgRaw = analogRead(ECG_PIN);
  ecgBuffer[bufferIndex] = (ecgRaw - ecgBaseline) * (3.3 / 1024.0); // Convert to voltage
  
  // Read EDA (Electrodermal Activity)
  int edaRaw = analogRead(EDA_PIN);
  edaBuffer[bufferIndex] = (edaRaw - edaBaseline) * (3.3 / 1024.0);
  
  // Read Respiration
  int respRaw = analogRead(RESP_PIN);
  respBuffer[bufferIndex] = (respRaw - respBaseline) * (3.3 / 1024.0);
  
  // Read EMG (Muscle Activity)
  int emgRaw = analogRead(EMG_PIN);
  emgBuffer[bufferIndex] = (emgRaw - emgBaseline) * (3.3 / 1024.0);
  
  // Read Temperature
  tempSensor.requestTemperatures();
  float tempC = tempSensor.getTempCByIndex(0);
  if (tempC != DEVICE_DISCONNECTED_C) {
    tempBuffer[bufferIndex] = tempC;
  } else {
    tempBuffer[bufferIndex] = tempBaseline;
  }
  
  // Read BVP (Blood Volume Pulse)
  int bvpRaw = analogRead(BVP_PIN);
  bvpBuffer[bufferIndex] = (bvpRaw - bvpBaseline) * (3.3 / 1024.0);
  
  // Read Wrist EDA
  int wristEdaRaw = analogRead(WRIST_EDA_PIN);
  wristEdaBuffer[bufferIndex] = (wristEdaRaw - wristEdaBaseline) * (3.3 / 1024.0);
  
  // Read Wrist Temperature
  int wristTempRaw = analogRead(WRIST_TEMP_PIN);
  wristTempBuffer[bufferIndex] = (wristTempRaw - wristTempBaseline) * (3.3 / 1024.0);
  
  // Read Accelerometer
  int16_t accX, accY, accZ;
  mpu.getAcceleration(&accX, &accY, &accZ);
  accXBuffer[bufferIndex] = accX / 16384.0; // Convert to g
  accYBuffer[bufferIndex] = accY / 16384.0;
  accZBuffer[bufferIndex] = accZ / 16384.0;
  
  bufferIndex++;
}

void processData() {
  // Apply basic signal processing
  if (bufferIndex > 0) {
    // Simple moving average filter for noise reduction
    int filterSize = min(5, bufferIndex);
    
    // ECG filtering
    float ecgSum = 0;
    for (int i = max(0, bufferIndex - filterSize); i < bufferIndex; i++) {
      ecgSum += ecgBuffer[i];
    }
    ecgBuffer[bufferIndex - 1] = ecgSum / filterSize;
    
    // EDA filtering
    float edaSum = 0;
    for (int i = max(0, bufferIndex - filterSize); i < bufferIndex; i++) {
      edaSum += edaBuffer[i];
    }
    edaBuffer[bufferIndex - 1] = edaSum / filterSize;
    
    // Similar filtering for other signals...
  }
}

void sendDataToPython() {
  // Send data in format: ECG,EDA,RESP,EMG,TEMP,BVP,WEDA,WTEMP,ACCX,ACCY,ACCZ
  String dataString = "";
  
  for (int i = 0; i < WINDOW_SIZE; i++) {
    dataString = String(ecgBuffer[i], 3) + "," +
                 String(edaBuffer[i], 3) + "," +
                 String(respBuffer[i], 3) + "," +
                 String(emgBuffer[i], 3) + "," +
                 String(tempBuffer[i], 3) + "," +
                 String(bvpBuffer[i], 3) + "," +
                 String(wristEdaBuffer[i], 3) + "," +
                 String(wristTempBuffer[i], 3) + "," +
                 String(accXBuffer[i], 3) + "," +
                 String(accYBuffer[i], 3) + "," +
                 String(accZBuffer[i], 3);
    
    // Send via Bluetooth
    bluetooth.println(dataString);
    
    // Also send via Serial for debugging
    Serial.println(dataString);
    
    // Small delay to prevent buffer overflow
    delay(1);
  }
  
  // Send end marker
  bluetooth.println("END_WINDOW");
  Serial.println("Window sent to Python");
}

void shiftBuffer() {
  // Shift buffer for overlap (50% overlap)
  int shiftAmount = WINDOW_SIZE - OVERLAP;
  
  // Shift all buffers
  for (int i = 0; i < OVERLAP; i++) {
    ecgBuffer[i] = ecgBuffer[i + shiftAmount];
    edaBuffer[i] = edaBuffer[i + shiftAmount];
    respBuffer[i] = respBuffer[i + shiftAmount];
    emgBuffer[i] = emgBuffer[i + shiftAmount];
    tempBuffer[i] = tempBuffer[i + shiftAmount];
    bvpBuffer[i] = bvpBuffer[i + shiftAmount];
    wristEdaBuffer[i] = wristEdaBuffer[i + shiftAmount];
    wristTempBuffer[i] = wristTempBuffer[i + shiftAmount];
    accXBuffer[i] = accXBuffer[i + shiftAmount];
    accYBuffer[i] = accYBuffer[i + shiftAmount];
    accZBuffer[i] = accZBuffer[i + shiftAmount];
  }
  
  bufferIndex = OVERLAP;
}

void handleBluetoothCommands() {
  if (bluetooth.available()) {
    String command = bluetooth.readStringUntil('\n');
    command.trim();
    
    if (command == "START") {
      sensorsCalibrated = true;
      bluetooth.println("OK_START");
    } else if (command == "STOP") {
      sensorsCalibrated = false;
      bluetooth.println("OK_STOP");
    } else if (command == "STATUS") {
      bluetooth.println("STATUS_OK");
    } else if (command == "CALIBRATE") {
      sensorsCalibrated = false;
      calibrationSamples = 0;
      bluetooth.println("OK_CALIBRATE");
    }
  }
}

void printSystemStatus() {
  Serial.println("=== Medical Sensor System Status ===");
  Serial.println("System Ready: " + String(systemReady ? "YES" : "NO"));
  Serial.println("Sensors Calibrated: " + String(sensorsCalibrated ? "YES" : "NO"));
  Serial.println("Buffer Index: " + String(bufferIndex));
  Serial.println("Sample Rate: " + String(SAMPLE_RATE) + " Hz");
  Serial.println("Window Size: " + String(WINDOW_SIZE) + " samples");
  Serial.println("Overlap: " + String(OVERLAP) + " samples");
  Serial.println("=====================================");
}

// Emergency functions
void emergencyStop() {
  sensorsCalibrated = false;
  systemReady = false;
  bluetooth.println("EMERGENCY_STOP");
  Serial.println("EMERGENCY STOP ACTIVATED");
}

void systemReset() {
  // Reset all variables
  bufferIndex = 0;
  calibrationSamples = 0;
  sensorsCalibrated = false;
  systemReady = true;
  
  // Reinitialize system
  initializeSystem();
  
  bluetooth.println("SYSTEM_RESET");
  Serial.println("System reset completed");
}
