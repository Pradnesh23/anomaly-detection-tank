"""
=============================================================================
STEP 7: EXTERNAL DATASETS + SYNTHETIC DATA GENERATOR
=============================================================================
Two things this script does:

A) EXTERNAL DATASET GUIDE
   Lists Kaggle/UCI/GitHub datasets compatible with your project,
   explains how to merge them with your HC-SR04 data, and what
   preprocessing adjustments are needed for each.

B) SYNTHETIC ANOMALY GENERATOR
   Creates labeled anomaly data to augment your 998-minute dataset.
   Generates: slow_leak, sudden_drain, sensor_spike, overflow, refill.
   Output: data/synthetic_labeled.csv (can be merged with your real data)

Writes: data/synthetic_labeled.csv
        data/combined_dataset.csv     (real + synthetic, labeled)
=============================================================================
"""

import pandas as pd
import numpy as np
import os
os.makedirs("data", exist_ok=True)

# -------------------------------------------------------------
# PART A: EXTERNAL DATASET GUIDE
# -------------------------------------------------------------
DATASET_GUIDE = """
====================================================================
|            COMPATIBLE PUBLIC DATASETS FOR THIS PROJECT          |
====================================================================

Your data: time-series of distance_mm (proxy for liquid level)
           ~3 readings/min, 16.6 hours, 1 HC-SR04 sensor

All datasets below share the same fundamental structure:
univariate/multivariate time-series with labeled or unlabeled anomalies.

-----------------------------------------------------------------
1. SKAB -- Skoltech Anomaly Benchmark (RECOMMENDED #1)
-----------------------------------------------------------------
URL    : https://github.com/waico/SKAB
         https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab
Domain : Water pump + valve IoT sensor data (PHYSICAL ANALOG to your tank)
Format : CSV, multiple files, datetime index, labeled anomalies
Columns: datetime, Accelerometer1RMS, Accelerometer2RMS, Current,
         Pressure, Temperature, Vibration, anomaly (0/1), changepoint
Why    : Same physical domain (water system), same IoT sensor setup,
         has ground truth labels -- directly usable for F1 evaluation
Size   : 34 files, ~35,000 rows total
Merge  : Normalize 'Pressure' column to [0,1] same as your level_pct.
         Extract same features: rolling_mean, RoC, std.
         Train Isolation Forest jointly on SKAB (labeled) + your data.

Preprocessing for SKAB:
  df_skab = pd.read_csv('valve1/1.csv', sep=';', index_col='datetime', parse_dates=True)
  df_skab['value'] = df_skab['Pressure']  # use pressure as level proxy
  # df_skab['anomaly'] already 0/1 -- use as ground truth

-----------------------------------------------------------------
2. PUMP SENSOR DATA (RECOMMENDED #2)
-----------------------------------------------------------------
URL    : https://www.kaggle.com/datasets/nphantawee/pump-sensor-data
Domain : Industrial pump with 52 sensors, machine failure labels
Format : 220,320 rows, datetime index, 'machine_status' column
         (NORMAL / BROKEN / RECOVERING)
Why    : Large labeled dataset for training ML classifiers.
         'BROKEN' status = anomaly ground truth.
         Use sensor_00 (pressure/flow) as your primary feature.
Size   : ~52MB, 220K rows
Merge  : Extract sensor_00 (most correlated to liquid level).
         Use 'machine_status' == 'BROKEN' as anomaly label.
         Downsample to 1-min resolution to match your data.

Preprocessing:
  df_pump = pd.read_csv('sensor.csv', parse_dates=['timestamp'])
  df_pump = df_pump.set_index('timestamp').resample('1min').mean()
  df_pump['anomaly'] = (df_pump['machine_status'] == 'BROKEN').astype(int)
  df_pump['distance_mm'] = df_pump['sensor_00'] * 300  # scale to mm range

-----------------------------------------------------------------
3. NUMENTA ANOMALY BENCHMARK (NAB)
-----------------------------------------------------------------
URL    : https://github.com/numenta/NAB
         https://www.kaggle.com/datasets/boltzmannbrain/nab
Domain : IoT machine temperature, AWS server metrics, traffic flow
Format : CSV per dataset, timestamp + value + anomaly_score
Why    : Industry-standard anomaly benchmark used in 200+ papers.
         Compare your algorithm F1 scores against published baselines.
         'realKnownCause/machine_temperature_system_failure.csv'
         is the closest to your hardware anomaly scenario.
Size   : 58 datasets, ~200-5000 rows each
Merge  : Use as-is. Normalize 'value' to your distance_mm scale.
         Labels from NAB JSON file (labels/combined_windows.json).

-----------------------------------------------------------------
4. WATER LEVEL SENSOR DATA (DIRECT MATCH)
-----------------------------------------------------------------
URL    : https://www.kaggle.com/datasets/srinuti/residential-water-usage-in-la-timeseries
         https://www.kaggle.com/search?q=water+level+sensor+time+series
Domain : Municipal water usage time series (daily/hourly)
Format : CSV, timestamp + water_level or usage_gallons
Why    : Same physical phenomenon -- water consumption creates the
         same gradual-drain pattern as your overnight readings.
         Good for training Prophet seasonal model.
Note   : Most water datasets are daily resolution -- resample or
         use only for Prophet training (seasonality patterns).

-----------------------------------------------------------------
5. YAHOO S5 ANOMALY DATASET
-----------------------------------------------------------------
URL    : https://webscope.sandbox.yahoo.com/catalog.php?datatype=s
Domain : Server metrics, synthetic anomalies (point + contextual)
Format : CSV, timestamp + value + is_anomaly (0/1)
Why    : A1/A2 benchmarks have synthetic anomalies that match your
         categories: point anomaly (sensor spike) + trend anomaly (slow leak).
         Can validate your detector on known ground truth.
Note   : Requires Yahoo account registration (free).

-----------------------------------------------------------------
HOW TO USE EXTERNAL DATASETS WITH YOUR PROJECT
-----------------------------------------------------------------
Option 1 -- Transfer Learning for Isolation Forest:
  Train IF on large external dataset (SKAB/pump data).
  Fine-tune (continue fit) on your data.
  Use external labels to calibrate the contamination parameter.

Option 2 -- F1 Validation:
  Apply your algorithm pipeline (MA+SD, RoC, CUSUM, IF) to SKAB.
  Compute Precision/Recall/F1 against SKAB ground truth labels.
  Report these numbers in the paper -- they are reproducible.

Option 3 -- Prophet Seasonality:
  Train Prophet on a 30-day external water level dataset.
  Transfer the daily seasonality component to your data.
  This gives much better seasonal modeling than 1-day training.

Option 4 -- LSTM Pretraining:
  Pretrain LSTM Autoencoder on pump sensor data (220K rows).
  Fine-tune on your 998 rows. Dramatically reduces training data need.
"""

print(DATASET_GUIDE)

# -------------------------------------------------------------
# PART B: SYNTHETIC ANOMALY DATA GENERATOR
# -------------------------------------------------------------
print("\n" + "="*65)
print("SYNTHETIC ANOMALY GENERATOR")
print("="*65)
print("Generating labeled synthetic data to augment your real dataset...")
print("Why: Your real data has ~998 rows and unknown anomaly labels.")
print("Synthetic data provides:")
print("  1. Known ground truth labels for F1 evaluation")
print("  2. More training data for supervised classifiers")
print("  3. Representation of rare anomaly types")

np.random.seed(42)

def generate_normal_signal(n_minutes, start_distance=55, drain_rate=8.24/60,
                            noise_std=1.5):
    """
    Simulate realistic HC-SR04 tank readings.
    drain_rate: mm/minute (calibrated from your real data: 8.24mm/hour)
    noise_std: 1.5mm (slightly above real noise floor of 1.12mm)
    """
    t = np.arange(n_minutes)
    # Gradual drain
    trend = start_distance + drain_rate * t
    # Add sensor noise
    noise = np.random.normal(0, noise_std, n_minutes)
    # Add slight oscillation (pump cycles)
    oscillation = 0.5 * np.sin(2 * np.pi * t / 60)
    return np.clip(trend + noise + oscillation, 10, 290)


def inject_anomaly(signal, anomaly_type, start_idx, duration=20):
    """Inject a specific anomaly pattern into a copy of the signal."""
    s = signal.copy()
    end_idx = min(start_idx + duration, len(s))

    if anomaly_type == 'slow_leak':
        # Faster drain rate for 'duration' minutes
        extra_drain = np.linspace(0, 15, end_idx - start_idx)
        s[start_idx:end_idx] += extra_drain

    elif anomaly_type == 'sudden_drain_theft':
        # Step drop of 30-50mm over 2-3 minutes
        drop = np.random.uniform(30, 50)
        s[start_idx:start_idx+3] += np.linspace(0, drop, 3)
        s[start_idx+3:] += drop  # level stays lower

    elif anomaly_type == 'sensor_spike':
        # 1-2 isolated readings far from baseline
        spike_mag = np.random.uniform(20, 60)
        s[start_idx]   += spike_mag * np.random.choice([-1, 1])
        if start_idx + 1 < len(s):
            s[start_idx+1] += spike_mag * 0.3 * np.random.choice([-1, 1])

    elif anomaly_type == 'overflow':
        # Distance drops near zero (tank full / overflowed)
        s[start_idx:end_idx] = np.random.uniform(5, 25, end_idx - start_idx)

    elif anomaly_type == 'sensor_freeze':
        # Readings stuck at constant value (sensor malfunction)
        frozen_val = s[start_idx - 1]
        s[start_idx:end_idx] = frozen_val + np.random.normal(0, 0.05, end_idx - start_idx)

    elif anomaly_type == 'refill_event':
        # Rapid rise (distance drops fast -- tank being filled)
        fill_amount = np.random.uniform(40, 100)
        s[start_idx:start_idx+5] -= np.linspace(0, fill_amount, 5)
        s[start_idx+5:end_idx]   -= fill_amount
        s = np.clip(s, 10, 290)

    return np.clip(s, 10, 290)


# Generate synthetic dataset
TOTAL_MINUTES = 2000  # ~33 hours of synthetic data
anomaly_types = ['slow_leak', 'sudden_drain_theft', 'sensor_spike',
                 'overflow', 'sensor_freeze', 'refill_event']

# Place anomalies at well-spaced intervals
anomaly_schedule = [
    ('slow_leak',           200, 40),
    ('sudden_drain_theft',  400,  5),
    ('sensor_spike',        600,  2),
    ('overflow',            800, 15),
    ('sensor_freeze',      1000, 25),
    ('refill_event',       1200,  8),
    ('slow_leak',          1400, 50),
    ('sudden_drain_theft', 1600,  4),
    ('sensor_spike',       1700,  1),
    ('refill_event',       1800, 10),
]

# Generate base normal signal
signal = generate_normal_signal(TOTAL_MINUTES)
labels = np.zeros(TOTAL_MINUTES, dtype=int)   # 0 = normal
classes = np.array(['normal'] * TOTAL_MINUTES, dtype=object)

# Inject all anomalies
for atype, start, duration in anomaly_schedule:
    signal = inject_anomaly(signal, atype, start, duration)
    labels[start:start+duration] = 1
    classes[start:start+duration] = atype

# Build DataFrame
timestamps = pd.date_range('2025-10-12 00:00', periods=TOTAL_MINUTES, freq='1min')
df_syn = pd.DataFrame({
    'timestamp'      : timestamps,
    'distance_mm_raw': signal,
    'distance_mm'    : pd.Series(signal).rolling(10, min_periods=1).mean().values,
    'anomaly_label'  : labels,
    'anomaly_class'  : classes,
    'source'         : 'synthetic'
})

# Add computed features
df_syn['roc_1']     = df_syn['distance_mm'].diff().fillna(0)
df_syn['roll_std_10']= df_syn['distance_mm'].rolling(10, min_periods=1).std().fillna(0)
df_syn['level_pct'] = ((250 - df_syn['distance_mm']) / 250 * 100).clip(0, 100)

df_syn.to_csv('data/synthetic_labeled.csv', index=False)

print(f"\n  Generated {TOTAL_MINUTES} synthetic readings ({TOTAL_MINUTES/60:.1f} hours)")
print(f"  Normal readings   : {(labels==0).sum()}")
print(f"  Anomaly readings  : {(labels==1).sum()}")
print(f"  Anomaly breakdown :")
for atype in anomaly_types:
    cnt = (classes == atype).sum()
    print(f"    {atype:<25}: {cnt} readings")
print(f"\n  [OK] Saved: data/synthetic_labeled.csv")

# -------------------------------------------------------------
# MERGE WITH REAL DATA
# -------------------------------------------------------------
print("\nMerging synthetic + real data...")
try:
    df_real = pd.read_csv('data/processed.csv', index_col='timestamp', parse_dates=True)
    df_real = df_real.reset_index()
    df_real['anomaly_label'] = -1   # -1 = unknown (not labeled)
    df_real['anomaly_class'] = 'unknown'
    df_real['source']        = 'real'

    # Align columns
    common = ['timestamp', 'distance_mm', 'roc_1', 'roll_std_10',
              'level_pct', 'anomaly_label', 'anomaly_class', 'source']
    df_syn_common  = df_syn.rename(columns={'roc_1': 'roc_1'})[common]
    df_real_common = df_real.rename(columns={'roc_1': 'roc_1'})[
        [c for c in common if c in df_real.columns]
    ]
    for c in common:
        if c not in df_real_common.columns:
            df_real_common[c] = None

    df_combined = pd.concat([df_real_common, df_syn_common], ignore_index=True)
    df_combined.to_csv('data/combined_dataset.csv', index=False)
    print(f"  [OK] Combined dataset: {len(df_combined)} rows -> data/combined_dataset.csv")
    print(f"    Real (unlabeled) : {(df_combined['source']=='real').sum()} rows")
    print(f"    Synthetic (labeled): {(df_combined['source']=='synthetic').sum()} rows")
    print(f"\n  USE THIS FOR:")
    print(f"    - Training Isolation Forest on real + synthetic combined")
    print(f"    - Training/evaluating Random Forest classifier on synthetic labels")
    print(f"    - Computing Precision/Recall/F1 on synthetic portion")
except Exception as e:
    print(f"  (Could not merge -- run 01_preprocessing.py first): {e}")

print(f"\n{'='*65}")
print("DATASET GUIDE + SYNTHETIC GENERATION COMPLETE")
print("="*65)
