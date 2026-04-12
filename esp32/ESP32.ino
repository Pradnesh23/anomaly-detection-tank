/*
 * ============================================
 * ESP32 Distance Sensor - IoT Project
 * Flask Backend Integration with CSV Storage
 * Real-time Dashboard Updates
 * ============================================
 * 
 * Hardware:
 * - ESP32 Development Board
 * - HC-SR04 Ultrasonic Sensor
 * - OLED Display (128x64, I2C, SSD1306)
 * 
 * Features:
 * - WiFi connectivity
 * - Real-time distance measurement in MILLIMETERS (mm)
 * - HTTP POST to Flask server every 20 seconds
 * - OLED display with status
 * - LED indicator for close objects
 * - Live dashboard updates at http://YOUR_SERVER_IP:8000/dashboard
 * 
 * Date: 10/10/2025
 * Modified: Distance now in millimeters (mm)
 * ============================================
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <HTTPClient.h>

// ============================================
// INCLUDE CREDENTIALS
// ============================================
#include "credentials.h"

// ============================================
// WiFi Configuration (from credentials.h)
// ============================================
const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;

// Build server URL from credentials
String serverUrlBase = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/sensor";
const char* serverUrl = serverUrlBase.c_str();

// Dashboard URL for display
String dashboardUrl = String("http://") + SERVER_IP + ":" + SERVER_PORT + "/dashboard";

// ============================================
// OLED Display Configuration
// ============================================
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDRESS 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ============================================
// ESP32 Pin Definitions
// ============================================
#define TRIGGER_PIN 5      // GPIO 5 - Ultrasonic Trigger
#define ECHO_PIN 18        // GPIO 18 - Ultrasonic Echo
#define LED_PIN 2          // GPIO 2 - Built-in LED

// ============================================
// Global Variables
// ============================================
long duration;
float distance_mm;  // Now in millimeters with decimals
bool wifiConnected = false;
unsigned long lastSendTime = 0;
const unsigned long sendInterval = 20000;  // 20 seconds - matches dashboard refresh
int requestCount = 0;
int successCount = 0;
int failureCount = 0;

// ============================================
// Function: Connect to WiFi
// ============================================
void connectToWiFi() {
  Serial.println("\n" + String('=', 50));
  Serial.println("Attempting WiFi Connection...");
  Serial.println(String('=', 50));
  
  // Display on OLED
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Connecting WiFi...");
  display.setCursor(0, 15);
  display.print("SSID: ");
  display.println(ssid);
  display.display();
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
    
    if (attempts % 3 == 0) {
      display.setCursor(0, 35);
      display.print("Waiting");
      for(int i = 0; i < (attempts/3) % 4; i++) {
        display.print(".");
      }
      display.println("   ");
      display.display();
    }
  }
  
  Serial.println();
  
  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;
    
    Serial.println("✓ WiFi Connected Successfully!");
    Serial.print("✓ IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("✓ Signal Strength: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
    Serial.println("\n📊 Dashboard URL:");
    Serial.println("   " + dashboardUrl);
    Serial.println(String('=', 50) + "\n");
    
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("WiFi Connected!");
    display.drawLine(0, 12, 128, 12, SSD1306_WHITE);
    display.setCursor(0, 18);
    display.print("IP: ");
    display.println(WiFi.localIP());
    display.setCursor(0, 32);
    display.println("Dashboard:");
    display.setCursor(0, 42);
    display.print(SERVER_IP);
    display.print(":");
    display.println(SERVER_PORT);
    display.setCursor(0, 52);
    display.println("/dashboard");
    display.display();
    delay(3000);
  } else {
    wifiConnected = false;
    Serial.println("✗ WiFi Connection Failed!");
    
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("WiFi Failed!");
    display.display();
    delay(5000);
  }
}

// ============================================
// Function: Measure Distance (in millimeters)
// ============================================
int measureDistance() {
  digitalWrite(TRIGGER_PIN, LOW);
  delayMicroseconds(2);
  
  digitalWrite(TRIGGER_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGGER_PIN, LOW);
  
  duration = pulseIn(ECHO_PIN, HIGH, 30000);
  
  if (duration == 0) {
    return -1;
  }
  
  // Calculate distance in millimeters
  // Speed of sound = 340 m/s = 0.34 mm/µs
  // Distance = (duration * 0.34) / 2
  distance_mm = (duration * 0.34) / 2;
  
  // Valid range: 20mm to 4000mm (2cm to 400cm)
  if (distance_mm < 20 || distance_mm > 4000) {
    return -1;
  }
  
  return distance_mm;
}

// ============================================
// Function: Send Data to Flask Server
// ============================================
void sendDataToFlaskServer(int dist_mm) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n⚠️  WiFi Disconnected - Reconnecting...");
    wifiConnected = false;
    connectToWiFi();
    return;
  }
  
  HTTPClient http;
  requestCount++;
  
  Serial.println("\n" + String('-', 50));
  Serial.println("📤 Sending Data to Flask Server");
  Serial.println(String('-', 50));
  
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(5000);
  
  // Convert mm to cm for Flask server (which expects cm and converts to mm)
  float dist_cm = dist_mm / 10.0;
  
  // Simplified payload matching the new anomaly detection backend
  String jsonPayload = "{";
  jsonPayload += "\"distance_mm\":" + String(dist_mm) + ",";
  jsonPayload += "\"device_id\":\"" + String(DEVICE_NAME) + "\"";
  jsonPayload += "}";
  
  Serial.println("📦 Payload: " + jsonPayload);
  Serial.println("🎯 Target: " + String(serverUrl));
  
  int httpResponseCode = http.POST(jsonPayload);
  
  if (httpResponseCode > 0) {
    String response = http.getString();
    successCount++;
    
    Serial.println("✓ SUCCESS!");
    Serial.print("✓ HTTP Status: ");
    Serial.println(httpResponseCode);
    Serial.print("✓ Response: ");
    Serial.println(response);
    Serial.print("✓ Success Rate: ");
    Serial.print(successCount);
    Serial.print("/");
    Serial.print(requestCount);
    Serial.print(" (");
    Serial.print((successCount * 100.0) / requestCount, 1);
    Serial.println("%)");
    Serial.println("📊 Dashboard updating in ~15s");
  } else {
    failureCount++;
    Serial.println("✗ FAILED!");
    Serial.print("✗ Error Code: ");
    Serial.println(httpResponseCode);
    Serial.println("✗ Check server is running:");
    Serial.println("   python server.py");
  }
  
  Serial.println(String('-', 50));
  http.end();
}

// ============================================
// Function: Update OLED Display
// ============================================
void updateDisplay(int dist_mm, bool sending) {
  display.clearDisplay();
  
  // Status bar at top
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  
  if (wifiConnected && WiFi.status() == WL_CONNECTED) {
    display.print("WiFi");
    // Draw small WiFi icon
    display.setCursor(30, 0);
    display.print("OK");
  } else {
    display.print("WiFi ERR");
  }
  
  // Request counter
  display.setCursor(70, 0);
  display.print(successCount);
  display.print("/");
  display.print(requestCount);
  
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);
  
  // Distance label
  display.setTextSize(1);
  display.setCursor(0, 15);
  display.println("Distance:");
  
  // Display distance value
  display.setCursor(5, 30);
  
  if (dist_mm > 0 && dist_mm < 4000) {
    // For values < 1000mm, show in mm
    if (dist_mm < 1000) {
      display.setTextSize(2);
      display.print(dist_mm);
      display.setTextSize(1);
      display.setCursor(80, 38);
      display.print("mm");
    } else {
      // For values >= 1000mm, show in cm for better readability
      display.setTextSize(2);
      display.print(dist_mm / 10);
      display.setTextSize(1);
      display.setCursor(80, 38);
      display.print("cm");
    }
  } else {
    display.setTextSize(2);
    display.print("---");
  }
  
  // Status line at bottom
  display.setTextSize(1);
  display.setCursor(0, 56);
  
  if (sending) {
    display.print("Sending");
    // Animated dots
    for(int i = 0; i < (millis() / 300) % 4; i++) {
      display.print(".");
    }
  } else {
    unsigned long timeToNext = (sendInterval - (millis() - lastSendTime)) / 1000;
    if (timeToNext < sendInterval / 1000) {
      display.print("Next: ");
      display.print(timeToNext);
      display.print("s");
    } else {
      display.print("Ready");
    }
  }
  
  display.display();
}

// ============================================
// Setup Function
// ============================================
void setup() {
  Serial.begin(9600);
  delay(1000);
  
  Serial.println("\n\n");
  Serial.println(String('=', 60));
  Serial.println("ESP32 IoT DISTANCE SENSOR");
  Serial.println("Flask Backend + Real-time Dashboard");
  Serial.println(String('=', 60));
  Serial.println("Project: IoT Distance Sensor");
  Serial.println("Backend: Flask Server (Port 8000)");
  Serial.println("Storage: CSV File (Real-time)");
  Serial.println("Dashboard: Auto-refresh every 15s");
  Serial.println("Interval: 20 seconds");
  Serial.println("Unit: Millimeters (mm)");
  Serial.println(String('=', 60) + "\n");
  
  // Initialize pins
  pinMode(TRIGGER_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  
  Serial.println("✓ GPIO Pins Configured");
  
  // Initialize OLED
  if(!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    Serial.println("✗ OLED init failed!");
    for(;;);
  }
  
  Serial.println("✓ OLED Display Initialized\n");
  
  // Splash screen
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(25, 5);
  display.println("IoT");
  display.setCursor(10, 25);
  display.println("Distance");
  display.setCursor(15, 45);
  display.println("Sensor");
  display.setTextSize(1);
  display.setCursor(25, 57);
  display.println("v2.0 mm");
  display.display();
  delay(2000);
  
  // Connect to WiFi
  connectToWiFi();
  
  Serial.println("✓ Setup Complete!\n");
  Serial.println("📊 IMPORTANT: Open dashboard to view live data:");
  Serial.println("   " + dashboardUrl);
  Serial.println("\n💡 Dashboard features:");
  Serial.println("   - Real-time graph (last 15 readings)");
  Serial.println("   - Live data table");
  Serial.println("   - Auto-refresh every 15 seconds");
  Serial.println("   - Distance in millimeters (mm)");
  Serial.println(String('=', 60) + "\n");
  
  delay(1000);
}

// ============================================
// Main Loop Function
// ============================================
void loop() {
  // Measure distance in millimeters
  int currentDistance = measureDistance();
  bool validReading = (currentDistance > 0 && currentDistance < 4000);
  
  // Control LED based on distance (< 100mm = 10cm)
  if (validReading && currentDistance < 100) {
    digitalWrite(LED_PIN, HIGH);
  } else {
    digitalWrite(LED_PIN, LOW);
  }
  
  // Print to serial
  if (validReading) {
    Serial.print("📏 Distance: ");
    Serial.print(currentDistance);
    Serial.print(" mm (");
    Serial.print(currentDistance / 10.0, 1);
    Serial.print(" cm) | LED: ");
    Serial.print(currentDistance < 100 ? "ON " : "OFF");
    Serial.print(" | Next send: ");
    unsigned long timeRemaining = (sendInterval - (millis() - lastSendTime)) / 1000;
    Serial.print(timeRemaining);
    Serial.println("s");
  } else {
    Serial.println("⚠️  Invalid reading");
  }
  
  // Update OLED display
  updateDisplay(currentDistance, false);
  
  // Send data at 20 second intervals
  unsigned long currentTime = millis();
  if (currentTime - lastSendTime >= sendInterval) {
    if (validReading) {
      updateDisplay(currentDistance, true);
      sendDataToFlaskServer(currentDistance);
      lastSendTime = currentTime;
      
      // Remind user about dashboard
      if (requestCount % 10 == 0) {
        Serial.println("\n💡 View live data: " + dashboardUrl + "\n");
      }
    } else {
      Serial.println("⚠️  Skipping send - invalid reading");
      lastSendTime = currentTime;
    }
  }
  
  delay(500);
}