# 🛢 IoT Liquid Tank Monitoring with Anomaly Detection

A comprehensive IoT-based anomaly detection framework for liquid tank monitoring using HC-SR04 ultrasonic sensor, ESP32 microcontroller, and a 5-method ensemble detection pipeline.

## 🏗 Architecture

```mermaid
graph TD
    subgraph "Tier 1: Sensing (Edge)"
        A["📶 HC-SR04 Sensor"] -->|"Distance (mm)"| B["⚡ ESP32 (TinyML)"]
        B -->|"MA+SD | RoC | CUSUM"| B
    end
    
    subgraph "Tier 2: Edge Server (Raspberry Pi)"
        B -->|"HTTP POST /sensor"| C["🌐 Flask Backend"]
        C --> D["🧠 Full 5-Method Ensemble"]
        D -->|"STL + Prophet"| D
        D --> E[("💽 SQLite Database")]
        D -.->|"Alert / Critical"| F["📩 SMS Notification"]
    end
    
    subgraph "Tier 3: UI"
        E --> G["💻 Real-Time Web Dashboard"]
    end
```

**Three-Tier System:**
- **Tier 1 (Sensing):** HC-SR04 + ESP32 with on-device statistical detection (~536 bytes RAM)
- **Tier 2 (Edge Server):** Raspberry Pi running Flask with full 5-method ensemble
- **Tier 3 (Visualization):** Real-time web dashboard with charts, alerts, and operator feedback

## 🔍 In-Depth System Architecture

```mermaid
graph LR
    %% Hardware Tier
    subgraph ESP32["ESP32 Microcontroller (C/C++)"]
        direction TB
        HC["HC-SR04 Pulse (GPIO)"] --> TinyML["TinyML Inference Engine"]
        TinyML --> MA["MA+SD (15μs)"]
        TinyML --> ROC["Rate of Change (5μs)"]
        TinyML --> CUSUM["Adaptive CUSUM (10μs)"]
        CUSUM --> WiFi["WiFi / HTTPClient"]
    end

    %% Python Backend Tier
    subgraph Server["Edge Server (Python/Flask)"]
        direction TB
        Route["POST /sensor"] --> Buffer["Rolling Window Buffer"]
        
        %% Ensemble Engine
        subgraph Ensemble["5-Method Ensemble Engine"]
            Buffer --> StatEngine["Statistical Evaluator"]
            Buffer --> TSAEngine["Time-Series Evaluator"]
            TSAEngine -.-> STL["STL Decomposition"]
            TSAEngine -.-> PROPHET["Prophet Forecasting"]
            StatEngine --> Voting["Confidence & Voting Layer"]
            TSAEngine --> Voting
        end
        
        Voting --> Flow["Flow/Usage Calculator"]
        
        %% Database
        subgraph DB["SQLite Database"]
            Readings[("readings (7.4k rows)")]
            Alerts[("alerts table")]
        end
        
        Flow --> DB
    end

    %% Client Tier
    subgraph Client["Web Dashboard (HTML/JS)"]
        direction TB
        Fetch["Polling (GET /status)"] --> DOM["DOM Elements"]
        DOM --> Charts["Chart.js Renderers"]
        DOM --> Log["Anomaly Data Table"]
        Log --> Operator["Operator Feedback ([OK] / [X])"]
    end

    %% Alerting out of band
    subgraph Notifications["External Services"]
        Twilio["Twilio SMS API"]
        Email["SMTP Email Server"]
    end

    %% Connections
    WiFi == "JSON Payload" ==> Route
    Voting -- "Score >= 0.7" --> Twilio
    Fetch == "JSON State" ==> DB
    Operator -- "POST /feedback" --> DB
```

## 🔄 User Flow & Data Lifecycle

```mermaid
sequenceDiagram
    participant S as HC-SR04 Sensor
    participant E as ESP32
    participant F as Flask (RPi)
    participant D as SQLite DB
    participant U as Dashboard (Browser)
    participant A as Alert System
    
    loop Every ~20 seconds
        S->>E: Ultrasonic pulse (distance)
        E->>F: POST /sensor {"distance_mm": 42.5}
        F->>F: Run MA+SD, RoC, CUSUM
        F->>F: Compute confidence score
        F->>D: INSERT INTO readings
        
        alt Score >= 0.4 (Alert/Critical)
            F->>D: INSERT INTO alerts
            F->>A: Trigger notification
            A-->>U: SMS / Email alert
        end
        
        F-->>E: JSON result (for local display)
    end
    
    loop Every 5 seconds
        U->>F: GET /status
        F->>D: SELECT latest reading
        D-->>F: Row data
        F-->>U: JSON (level, score, tier)
        U->>U: Update cards + charts
    end
    
    Note over U,D: Human-in-the-Loop Feedback
    U->>F: Operator clicks [CONFIRM] or [FALSE ALARM]
    F->>D: Update label for future training
```

## 📊 Detection Methods

| Method | Type | F1 Score | Inference | Edge Deployable |
|--------|------|----------|-----------|-----------------|
| MA+SD (Residual) | Statistical | 0.162 | ~15 μs | ESP32 ✅ |
| Rate of Change (Residual) | Statistical | 0.257 | ~5 μs | ESP32 ✅ |
| Adaptive CUSUM (Residual) | Statistical | 0.342 | ~10 μs | ESP32 ✅ |
| **Hybrid (3-method)** | **Ensemble** | **1.000** | **~30 μs** | **ESP32 ✅** |
| STL Decomposition | TSA | 0.376 | ~100 ms | RPi only |
| Prophet Forecast | TSA | 0.857 | ~200 ms | RPi only |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib statsmodels prophet scipy flask
```

### 2. Run the Full Pipeline
```bash
python ml_pipeline/run_pipeline.py
```
This executes 7 steps: definitions → preprocessing → statistical detectors → STL+Prophet → evaluation → synthetic data → TinyML export

### 3. Load Data & Start Dashboard
```bash
python backend/db_loader.py    # Load pipeline results into SQLite
python backend/app.py          # Start Flask server
```
Open: http://localhost:5000/dashboard

### 4. View Project Report
Open: http://localhost:5000/report

## 📁 Project Structure

```
tank_monitor/
├── esp32/
│   ├── ESP32.ino                # Arduino firmware sending to /sensor
│   ├── credentials.h            # WiFi configurations
│   └── tinyml_detectors.h       # C headers for ESP32 deployment
├── backend/
│   ├── app.py                   # Dashboard + REST API + real-time pipeline
│   ├── db_loader.py             # CSV → SQLite loader
│   └── simulator.py             # Synthetic ESP32 sensor simulator
├── ml_pipeline/
│   ├── 00_anomaly_definitions.py    # Formal anomaly taxonomy + thresholds
│   ├── 01_preprocessing.py          # Data cleaning + feature engineering
│   ├── 02_statistical_detectors.py  # MA+SD, Rate of Change, Adaptive CUSUM
│   ├── 03_ml_detectors.py           # STL Decomposition + Prophet
│   ├── 05_evaluation.py             # Metrics, SOTA comparison, plots
│   ├── 07_datasets_and_synthetic.py # Synthetic data generation
│   ├── 08_tinyml_export.py          # TinyML thresholds export
│   └── run_pipeline.py              # Execute all ML scripts
├── data/                        # Datasets + SQLite database
├── models/                      # Thresholds (shared)
└── output/                      # Reports, plots, project_report.html
```

## 📈 Key Results

- **7,417 processed readings** from 21K raw HC-SR04 measurements (123.6 hours)
- **Hybrid (3-method) Ensemble** achieves perfect F1=1.000 on real data
- **Zero False Alarms** for the Hybrid ensemble (0.00 FP/hour)
- **TinyML deployment** uses only **536 bytes** of ESP32 RAM
- **Sub-millisecond inference** (~32 μs total for 3 statistical detectors)

## 🔌 Real-Time Setup (ESP32 → RPi)

The Flask backend has a `POST /sensor` endpoint. ESP32 sends:
```json
{"distance_mm": 42.5, "device_id": "ESP32-Sensor-01"}
```

Flask runs MA+SD + RoC + CUSUM, stores in SQLite, and triggers alerts if anomaly detected.

## 📄 Publication

- **IEEE Conference Paper** — Accepted (addressing 9 reviewer improvement points)
- **Journal Extension** — Planned (TinyML edge deployment + field results)

## 🛠 Technologies

Python, Flask, Prophet, statsmodels (STL), scikit-learn, SQLite, Chart.js, TinyML (C), ESP32, HC-SR04
