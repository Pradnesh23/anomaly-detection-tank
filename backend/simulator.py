import csv
import time
import requests
import sys

csv_file = "data/combined_dataset.csv"
url = "http://localhost:5000/sensor"
delay = 0.5  # Time between readings (seconds)

print(f"Reading from {csv_file} and POSTing to {url}")
print("Press Ctrl+C to stop.\n")

try:
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            dist = float(row['distance_mm'])
            payload = {"distance_mm": dist, "device_id": "Simulated-ESP32"}
            
            try:
                resp = requests.post(url, json=payload)
                print(f"Reading {count+1}: Sent {dist:.2f} mm -> HTTP {resp.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Error sending data: {e}")
                
            time.sleep(delay)
            count += 1
            if count > 500:  # Safety limit
                print("\nReached 500 data points. Stopping.")
                break
except KeyboardInterrupt:
    print("\nSimulation stopped.")
except Exception as e:
    print(f"\nError reading dataset: {e}")
