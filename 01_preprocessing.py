"""
=============================================================================
STEP 1: DATA PREPROCESSING PIPELINE
=============================================================================
IoT Liquid Tank Monitoring -- HC-SR04 Single Sensor

Supports two datasets:
  - OLD: sensor_data_20251010-20251011.csv (998 rows, ~16 hours)
  - NEW: Thursday_Tuesday.csv (21,357 rows, 5 days)

What this script does:
  1. Load raw CSV & auto-detect format
  2. Remove init/bad rows
  3. Aggregate multiple readings per minute -> single value
  4. Reindex to uniform 1-minute grid
  5. Fill gaps safely
  6. Smooth sensor noise (rolling mean)
  7. Engineer all features for detection
  8. Generate dataset characteristics report (reviewer item #1)
  9. Save clean dataset

Writes: data/processed.csv
        output/dataset_report.txt

Addresses reviewer feedback:
  - "clearer details on sensor data characteristics" (R1 + R2)
  - "more comprehensive experiments" (R2) -- via 21K dataset support
=============================================================================
"""

import pandas as pd
import numpy as np
import os, warnings, json
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# -------------------------------------------------------------
# CONFIG -- Select which dataset to process
# -------------------------------------------------------------
DATASET_MODE = "new"   # "old", "new", or "both"

DATASETS = {
    "old": {
        "path": "data/sensor_data_20251010-20251011.csv",
        "timestamp_format": "%d-%m-%Y %H:%M",
        "valid_device_id": "ESP32-Sensor-01",
        "init_device_id":  "ESP32-Test",
        "description": "Original 998-row dataset (Oct 10-11, 2025, ~16 hours)"
    },
    "new": {
        "path": "data/Thursday_Tuesday.csv",
        "timestamp_format": "%Y-%m-%d %H:%M:%S",
        "valid_device_id": "ESP32_SENSOR_1",
        "init_device_id":  None,  # no init row in new dataset
        "description": "Extended 21K-row dataset (Mar 26-31, 2026, ~5 days)"
    }
}

OUTPUT_PATH    = "data/processed.csv"
TANK_HEIGHT_MM = 250   # physical tank height in mm

os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)

# -------------------------------------------------------------
# SENSOR SPECIFICATIONS (for dataset report)
# -------------------------------------------------------------
SENSOR_SPECS = {
    "model":           "HC-SR04 Ultrasonic Distance Sensor",
    "range":           "2 cm -- 400 cm",
    "accuracy":        "+-3 mm",
    "resolution":      "1 mm",
    "operating_voltage": "5V DC",
    "frequency":       "40 kHz",
    "trigger_pulse":   "10 us TTL",
    "measurement_angle": "~15deg",
    "microcontroller":   "ESP32 DevKit v1",
    "sampling_interval": "~20 seconds",
    "tank_height_mm":    TANK_HEIGHT_MM,
    "communication":     "HTTP POST over WiFi to Flask server"
}


def load_dataset(name: str) -> pd.DataFrame:
    """Load and parse a single dataset."""
    cfg = DATASETS[name]
    print(f"\n  Loading: {cfg['path']}")
    print(f"  Description: {cfg['description']}")

    df_raw = pd.read_csv(cfg['path'])
    print(f"  Raw rows loaded: {len(df_raw)}")
    print(f"  Columns: {df_raw.columns.tolist()}")

    # Drop rows with empty/NaN timestamps
    before_len = len(df_raw)
    df_raw = df_raw.dropna(subset=['timestamp'])
    df_raw = df_raw[df_raw['timestamp'].str.strip() != '']
    dropped_empty = before_len - len(df_raw)
    if dropped_empty > 0:
        print(f"  [!] Dropped {dropped_empty} rows with empty timestamps")

    # Parse timestamps
    df_raw['timestamp'] = pd.to_datetime(
        df_raw['timestamp'], format=cfg['timestamp_format'], errors='coerce'
    )
    bad_ts = df_raw['timestamp'].isna().sum()
    if bad_ts > 0:
        print(f"  [!] Dropped {bad_ts} rows with unparseable timestamps")
        df_raw = df_raw.dropna(subset=['timestamp'])

    # Remove init rows
    if cfg['init_device_id'] and cfg['init_device_id'] in df_raw['device_id'].values:
        init_count = (df_raw['device_id'] == cfg['init_device_id']).sum()
        print(f"  Removing {init_count} init row(s) (device_id={cfg['init_device_id']})")
        df_raw = df_raw[df_raw['device_id'] == cfg['valid_device_id']]
    elif cfg['valid_device_id']:
        df_raw = df_raw[df_raw['device_id'] == cfg['valid_device_id']]

    # Keep only needed columns
    df = df_raw[['timestamp', 'distance_mm']].copy()
    df['distance_mm'] = pd.to_numeric(df['distance_mm'], errors='coerce')
    df = df.dropna(subset=['distance_mm'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"  Clean rows: {len(df)}")
    print(f"  Time range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"  Distance range: {df['distance_mm'].min():.1f} -- {df['distance_mm'].max():.1f} mm")

    return df


def aggregate_to_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to 1-minute resolution and reindex to uniform grid."""
    # Floor timestamps to nearest minute
    df['timestamp_min'] = df['timestamp'].dt.floor('min')
    per_min = df.groupby('timestamp_min').size()

    print(f"\n  Readings per minute distribution:")
    for cnt in sorted(per_min.unique()):
        n = (per_min == cnt).sum()
        print(f"    {cnt} reading(s)/min: {n} minutes")

    # Aggregate: mean of all readings within same minute
    df_min = df.groupby('timestamp_min')['distance_mm'].mean().reset_index()
    df_min = df_min.rename(columns={'timestamp_min': 'timestamp', 'distance_mm': 'distance_mm_raw'})

    print(f"\n  Unique minutes with data: {len(df_min)}")

    # Reindex to uniform 1-minute grid
    full_range = pd.date_range(df_min['timestamp'].min(),
                               df_min['timestamp'].max(), freq='1min')
    df_min = df_min.set_index('timestamp').reindex(full_range)
    df_min.index.name = 'timestamp'

    missing_pts = df_min['distance_mm_raw'].isna().sum()
    total_slots = len(full_range)
    coverage_pct = (1 - missing_pts / total_slots) * 100

    print(f"  Total 1-minute slots: {total_slots}")
    print(f"  Slots with data: {total_slots - missing_pts}")
    print(f"  Missing slots (gaps): {missing_pts}")
    print(f"  Data coverage: {coverage_pct:.1f}%")

    # Gap analysis
    if missing_pts > 0:
        gap_mask = df_min['distance_mm_raw'].isna()
        gap_groups = (gap_mask != gap_mask.shift()).cumsum()
        gap_lengths = gap_mask.groupby(gap_groups).sum()
        gap_lengths = gap_lengths[gap_lengths > 0]

        print(f"\n  Gap distribution:")
        print(f"    Number of gaps: {len(gap_lengths)}")
        print(f"    Gap lengths (minutes): min={gap_lengths.min():.0f}, "
              f"max={gap_lengths.max():.0f}, mean={gap_lengths.mean():.1f}")

        # Fill strategy: linear for short gaps, forward fill for large
        df_min['distance_mm_raw'] = df_min['distance_mm_raw'].interpolate(
            method='linear', limit=5)
        remaining = df_min['distance_mm_raw'].isna().sum()
        if remaining > 0:
            print(f"    {remaining} gap(s) too large for interpolation -> forward fill")
            df_min['distance_mm_raw'] = df_min['distance_mm_raw'].ffill().bfill()

    print(f"  Final missing values: {df_min['distance_mm_raw'].isna().sum()}")
    return df_min


def smooth_and_derive(df_min: pd.DataFrame) -> pd.DataFrame:
    """Apply smoothing and compute derived columns."""
    # Noise floor measurement
    noise_std = (df_min['distance_mm_raw'] -
                 df_min['distance_mm_raw'].rolling(10).mean()).dropna().std()
    print(f"\n  Measured noise floor: +-{noise_std:.2f} mm")
    print(f"  HC-SR04 datasheet accuracy: +-3 mm")

    # Smoothing
    df_min['distance_mm'] = df_min['distance_mm_raw'].rolling(
        window=10, min_periods=1).mean()

    # Liquid level
    df_min['level_mm'] = TANK_HEIGHT_MM - df_min['distance_mm']
    df_min['level_pct'] = (df_min['level_mm'] / TANK_HEIGHT_MM * 100).clip(0, 100)

    return df_min, noise_std


def engineer_features(df_min: pd.DataFrame) -> pd.DataFrame:
    """Create all features for detection pipeline."""
    # Time features
    df_min['hour']          = df_min.index.hour
    df_min['minute_of_day'] = df_min.index.hour * 60 + df_min.index.minute
    df_min['is_night']      = ((df_min['hour'] >= 22) | (df_min['hour'] < 6)).astype(int)

    # Multi-day features (for new dataset)
    total_days = (df_min.index.max() - df_min.index.min()).total_seconds() / 86400
    if total_days > 1.5:
        df_min['day_of_week']  = df_min.index.dayofweek      # 0=Mon, 6=Sun
        df_min['is_weekend']   = (df_min['day_of_week'] >= 5).astype(int)
        df_min['day_number']   = (df_min.index - df_min.index[0]).days
        print(f"  Multi-day features added (span={total_days:.1f} days): "
              f"day_of_week, is_weekend, day_number")
    else:
        df_min['day_of_week']  = df_min.index.dayofweek
        df_min['is_weekend']   = 0
        df_min['day_number']   = 0

    # Rolling statistics
    for w in [5, 10, 30, 60]:
        df_min[f'roll_mean_{w}'] = df_min['distance_mm'].rolling(w, min_periods=1).mean()
        df_min[f'roll_std_{w}']  = df_min['distance_mm'].rolling(w, min_periods=1).std().fillna(0)

    # Rate of Change
    df_min['roc_1']  = df_min['distance_mm'].diff(1)
    df_min['roc_5']  = df_min['distance_mm'].diff(5)
    df_min['roc_10'] = df_min['distance_mm'].diff(10)

    # Deviation from rolling mean (residual -- this is what MA+SD uses)
    df_min['deviation_10'] = df_min['distance_mm'] - df_min['roll_mean_10']
    df_min['deviation_30'] = df_min['distance_mm'] - df_min['roll_mean_30']

    # Adaptive CUSUM features (using calibrated parameters from 00_anomaly_definitions)
    alpha = 0.15     # EWMA smoothing factor
    k_cusum = 0.5    # allowance
    h_cusum = 10.0   # detection threshold

    ewma_baseline = df_min['distance_mm'].iloc[0]
    ewma_vals, cusum_pos_vals, cusum_neg_vals = [], [], []
    cusum_pos, cusum_neg = 0.0, 0.0

    for val in df_min['distance_mm']:
        ewma_baseline = alpha * val + (1 - alpha) * ewma_baseline
        cusum_pos = max(0, cusum_pos + (val - ewma_baseline - k_cusum))
        cusum_neg = max(0, cusum_neg - (val - ewma_baseline + k_cusum))
        ewma_vals.append(ewma_baseline)
        cusum_pos_vals.append(cusum_pos)
        cusum_neg_vals.append(cusum_neg)
        if cusum_pos > h_cusum or cusum_neg > h_cusum:
            cusum_pos, cusum_neg = 0.0, 0.0  # reset after detection

    df_min['ewma_baseline']   = ewma_vals
    df_min['cusum_pos']       = cusum_pos_vals
    df_min['cusum_neg']       = cusum_neg_vals
    df_min['cusum_deviation'] = df_min['distance_mm'] - df_min['ewma_baseline']

    # Z-score
    df_min['zscore_10'] = (df_min['deviation_10'] /
                           df_min['roll_std_10'].replace(0, np.nan)).fillna(0)

    # Lag features
    for lag in [1, 2, 5, 10]:
        df_min[f'lag_{lag}'] = df_min['distance_mm'].shift(lag)

    # Fill NaN lags
    df_min = df_min.ffill().bfill()

    return df_min


def generate_dataset_report(df_min: pd.DataFrame, noise_std: float, dataset_name: str):
    """
    Generate comprehensive dataset characteristics report.
    Addresses reviewer feedback: "clearer details on sensor data characteristics"
    """
    report_lines = []

    def p(line=""):
        report_lines.append(line)
        print(line)

    p("=" * 70)
    p("DATASET CHARACTERISTICS REPORT")
    p("For inclusion in paper Section: Dataset Description")
    p("=" * 70)

    p(f"\nDataset: {dataset_name}")
    p(f"Total data points: {len(df_min)}")
    p(f"Time range: {df_min.index.min()} -> {df_min.index.max()}")
    total_hours = (df_min.index.max() - df_min.index.min()).total_seconds() / 3600
    total_days  = total_hours / 24
    p(f"Duration: {total_hours:.1f} hours ({total_days:.1f} days)")

    p(f"\n{'-' * 70}")
    p("SENSOR SPECIFICATIONS")
    p(f"{'-' * 70}")
    for key, val in SENSOR_SPECS.items():
        p(f"  {key:25s}: {val}")

    p(f"\n{'-' * 70}")
    p("SAMPLING CHARACTERISTICS")
    p(f"{'-' * 70}")
    diffs = pd.Series(df_min.index).diff().dropna()
    diffs_sec = diffs.dt.total_seconds()
    p(f"  Grid resolution      : 1 minute (aggregated from ~20s raw)")
    p(f"  Total 1-min slots    : {len(df_min)}")
    p(f"  Expected readings/min: ~3 (at 20s intervals)")

    p(f"\n{'-' * 70}")
    p("STATISTICAL PROFILE OF distance_mm")
    p(f"{'-' * 70}")
    dist = df_min['distance_mm']
    p(f"  Count    : {dist.count()}")
    p(f"  Mean     : {dist.mean():.2f} mm")
    p(f"  Std Dev  : {dist.std():.2f} mm")
    p(f"  Min      : {dist.min():.2f} mm")
    p(f"  25%      : {dist.quantile(0.25):.2f} mm")
    p(f"  Median   : {dist.median():.2f} mm")
    p(f"  75%      : {dist.quantile(0.75):.2f} mm")
    p(f"  Max      : {dist.max():.2f} mm")
    p(f"  Skewness : {dist.skew():.3f}")
    p(f"  Kurtosis : {dist.kurtosis():.3f}")

    p(f"\n{'-' * 70}")
    p("NOISE FLOOR ESTIMATION")
    p(f"{'-' * 70}")
    p(f"  Method: Std dev of (raw - rolling_mean_10)")
    p(f"  Measured noise floor : +-{noise_std:.2f} mm")
    p(f"  HC-SR04 spec accuracy: +-3.00 mm")
    p(f"  Noise is {'within' if noise_std <= 3.0 else 'EXCEEDING'} sensor specifications")

    # Noise from consecutive differences (alternative estimator)
    consec_diff = df_min['distance_mm_raw'].diff().dropna()
    consec_noise = consec_diff.std() / np.sqrt(2)  # divide by sqrt(2) for differenced noise
    p(f"  Alt. estimator (diff/sqrt2): +-{consec_noise:.2f} mm")

    p(f"\n{'-' * 70}")
    p("DATA QUALITY METRICS")
    p(f"{'-' * 70}")
    raw_nans = df_min['distance_mm_raw'].isna().sum()
    total_pts = len(df_min)
    # Duplicate check based on index
    dup_ts = df_min.index.duplicated().sum()
    # Out of range check (HC-SR04 range: 20-4000mm, practical: 20-250mm)
    oor = ((df_min['distance_mm_raw'] < 20) | (df_min['distance_mm_raw'] > TANK_HEIGHT_MM)).sum()
    p(f"  Missing data points  : {raw_nans} ({raw_nans/total_pts*100:.2f}%)")
    p(f"  Duplicate timestamps : {dup_ts}")
    p(f"  Out-of-range readings: {oor} (outside 20-{TANK_HEIGHT_MM}mm)")

    # Per-day summary (for multi-day datasets)
    n_days = (df_min.index.max() - df_min.index.min()).days + 1
    if n_days > 1:
        p(f"\n{'-' * 70}")
        p("PER-DAY SUMMARY")
        p(f"{'-' * 70}")
        p(f"  {'Date':<14s} {'Count':>7s} {'Mean':>8s} {'Std':>7s} {'Min':>7s} {'Max':>7s}")
        p(f"  {'-'*14} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
        for date, grp in df_min.groupby(df_min.index.date):
            d = grp['distance_mm']
            p(f"  {str(date):<14s} {len(grp):>7d} {d.mean():>8.2f} "
              f"{d.std():>7.2f} {d.min():>7.2f} {d.max():>7.2f}")

    p(f"\n{'=' * 70}")

    # Save report
    with open("output/dataset_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[OK] Saved: output/dataset_report.txt")


# -------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 65)
    print("STEP 1: DATA PREPROCESSING PIPELINE")
    print("=" * 65)
    print(f"\nDataset mode: {DATASET_MODE}")

    if DATASET_MODE == "both":
        datasets_to_run = ["old", "new"]
    else:
        datasets_to_run = [DATASET_MODE]

    for ds_name in datasets_to_run:
        cfg = DATASETS[ds_name]

        if not os.path.exists(cfg['path']):
            print(f"\n[!] Dataset not found: {cfg['path']} -- skipping")
            continue

        print(f"\n{'=' * 65}")
        print(f"PROCESSING DATASET: {ds_name.upper()}")
        print(f"{'=' * 65}")

        # 1. Load
        print(f"\n{'-' * 65}")
        print("STEP 1.1: LOADING RAW DATA")
        print(f"{'-' * 65}")
        print("Column decisions:")
        print("  timestamp    -> KEEP  (primary time index)")
        print("  distance_mm  -> KEEP  (the only sensor signal we need)")
        print("  led_status   -> DROP  (derived threshold output -- data leakage risk)")
        print("  device_id    -> DROP  (used only for filtering, no signal information)")

        df = load_dataset(ds_name)

        # 2. Aggregate
        print(f"\n{'-' * 65}")
        print("STEP 1.2: AGGREGATE TO 1-MINUTE RESOLUTION")
        print(f"{'-' * 65}")
        print("Strategy: take MEAN of all readings within the same minute.")
        print("  -> Reduces noise from burst readings")
        print("  -> Preserves true signal level (mean is unbiased for Gaussian noise)")

        df_min = aggregate_to_minutes(df)

        # 3. Smooth
        print(f"\n{'-' * 65}")
        print("STEP 1.3: SENSOR NOISE SMOOTHING")
        print(f"{'-' * 65}")

        df_min, noise_std = smooth_and_derive(df_min)

        # 4. Feature engineering
        print(f"\n{'-' * 65}")
        print("STEP 1.4: FEATURE ENGINEERING")
        print(f"{'-' * 65}")

        df_min = engineer_features(df_min)

        # Print feature summary
        print("\nFeatures created:")
        feature_groups = {
            "Raw/Smoothed signal" : ['distance_mm_raw', 'distance_mm', 'level_mm', 'level_pct'],
            "Time features"       : ['hour', 'minute_of_day', 'is_night',
                                     'day_of_week', 'is_weekend', 'day_number'],
            "Rolling stats"       : [c for c in df_min.columns if 'roll_' in c],
            "Rate of change"      : ['roc_1', 'roc_5', 'roc_10'],
            "Deviation/Residual"  : ['deviation_10', 'deviation_30', 'zscore_10'],
            "Adaptive CUSUM"      : ['ewma_baseline', 'cusum_pos', 'cusum_neg', 'cusum_deviation'],
            "Lag features"        : [c for c in df_min.columns if 'lag_' in c],
        }
        for grp, cols in feature_groups.items():
            existing = [c for c in cols if c in df_min.columns]
            print(f"  {grp:25s}: {existing}")

        # 5. Dataset characteristics report
        print(f"\n{'-' * 65}")
        print("STEP 1.5: DATASET CHARACTERISTICS REPORT")
        print(f"{'-' * 65}")

        generate_dataset_report(df_min, noise_std, ds_name)

        # 6. Save
        out_path = OUTPUT_PATH if len(datasets_to_run) == 1 else f"data/processed_{ds_name}.csv"
        df_min.to_csv(out_path)
        print(f"\n[OK] Processed dataset saved to: {out_path}")
        print(f"  Shape: {df_min.shape}")
        print(f"  Time range : {df_min.index.min()} -> {df_min.index.max()}")
        print(f"  Distance   : {df_min['distance_mm'].min():.1f} - "
              f"{df_min['distance_mm'].max():.1f} mm")
        print(f"  Level      : {df_min['level_pct'].min():.1f}% - "
              f"{df_min['level_pct'].max():.1f}%")

    # If running both, also save the new dataset as primary
    if DATASET_MODE == "both" and os.path.exists("data/processed_new.csv"):
        import shutil
        shutil.copy("data/processed_new.csv", OUTPUT_PATH)
        print(f"\n[OK] Primary output (data/processed.csv) set to NEW dataset")

    print(f"\n{'=' * 65}")
    print("PREPROCESSING COMPLETE -- ready for Step 2 (statistical detection)")
    print("=" * 65)
