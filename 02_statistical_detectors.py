"""
=============================================================================
STEP 2: STATISTICAL ANOMALY DETECTORS
=============================================================================
Implements MA+SD, RoC, Adaptive CUSUM -- all from the paper.
Adds confidence scoring and rule-based anomaly classification.

Auto-scales parameters based on dataset size:
  - Small dataset  (< 2000 rows): window=50, original thresholds
  - Large dataset  (>= 2000 rows): window=100, same thresholds

Reads:  data/processed.csv
Writes: data/statistical_results.csv
        models/stat_thresholds.json
=============================================================================
"""

import pandas as pd
import numpy as np
import json, os
import warnings
warnings.filterwarnings('ignore')

PROCESSED_PATH = "data/processed.csv"
OUTPUT_PATH    = "data/statistical_results.csv"
THRESHOLDS_PATH= "models/stat_thresholds.json"

os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)

# -------------------------------------------------------------
# LOAD
# -------------------------------------------------------------
df = pd.read_csv(PROCESSED_PATH, index_col='timestamp', parse_dates=True)
print(f"Loaded processed data: {df.shape}")

# Auto-scale window for dataset size
DATASET_SIZE = len(df)
MASD_WINDOW = 100 if DATASET_SIZE >= 2000 else 50
print(f"Dataset size: {DATASET_SIZE} -> MA+SD window auto-set to {MASD_WINDOW}")

# -------------------------------------------------------------
# DETECTOR 1: MA+SD
# -------------------------------------------------------------
def detect_masd(series, window=50, n_std=2.5):
    """
    Moving Average + Standard Deviation detector.
    
    CRITICAL FIX: Your tank has a steady drain trend (~8.24mm/hr).
    A simple rolling mean always LAGS behind a trending signal by
    ~window/2 timesteps, creating a systematic offset that exceeds
    1.5sigma almost continuously -- flagging the entire dataset as anomalous.
    
    Fix: Detrend the signal first (remove linear drain trend),
    then apply MA+SD on the RESIDUALS. This separates:
      - Normal drift  = linear drain trend (expected, not anomalous)
      - Anomaly       = deviation from the trend (unexpected)
    
    n_std raised to 2.5 (from 1.5) on detrended signal to match
    the noise floor after removing the trend component.
    """
    from scipy import stats as scipy_stats
    x = np.arange(len(series))
    
    # Remove linear drain trend
    slope, intercept, _, _, _ = scipy_stats.linregress(x, series.values)
    detrended = series - (slope * x + intercept)
    
    roll_mean = detrended.rolling(window, min_periods=window//2).mean()
    roll_std  = detrended.rolling(window, min_periods=window//2).std()
    
    deviation = (detrended - roll_mean).abs()
    upper = roll_mean + n_std * roll_std
    lower = roll_mean - n_std * roll_std
    flag  = deviation > n_std * roll_std
    score = (deviation / roll_std.replace(0, np.nan)).fillna(0)
    
    # Return bounds in original signal space for plotting
    upper_orig = upper + (slope * x + intercept)
    lower_orig = lower + (slope * x + intercept)
    trend_line = pd.Series(slope * x + intercept, index=series.index)
    
    return flag, score, trend_line, upper_orig, lower_orig

# -------------------------------------------------------------
# DETECTOR 2: Rate of Change
# -------------------------------------------------------------
def detect_roc(series, drop_threshold=-1.0, rise_threshold=1.0):
    """
    Flags sudden drops (drain/theft) or rises (fill/overflow).
    Thresholds in mm/min. -1/+1 is the most sensitive setting.
    Note: HC-SR04 noise floor is +-1.12 mm, so +-1.0 is right at
    the noise boundary -- fine-tuned for this specific sensor.
    """
    roc = series.diff(1)
    flag  = (roc < drop_threshold) | (roc > rise_threshold)
    score = roc.abs() / abs(drop_threshold)  # normalized
    return flag, score, roc

# -------------------------------------------------------------
# DETECTOR 3: Adaptive CUSUM
# -------------------------------------------------------------
def detect_adaptive_cusum(series, alpha=0.15, k=0.5, h=10.0):
    """
    CUSUM with EWMA adaptive baseline.
    
    CRITICAL FIX: alpha raised from 0.03 -> 0.15, h raised from 5.0 -> 10.0.
    With a steady drain trend, a slow alpha (0.03) means the baseline
    barely moves -- CUSUM accumulates the drain as 'drift' and fires
    continuously. A faster alpha (0.15) tracks the drain trend properly,
    reserving CUSUM sensitivity for genuine sudden deviations.
    h=10 raises the detection threshold to avoid noise-level triggers.
    """
    mu = series.iloc[0]
    flags, scores = [], []
    cp, cn = 0.0, 0.0
    mu_vals = []

    for val in series:
        mu = alpha * val + (1 - alpha) * mu
        cp = max(0, cp + (val - mu - k))
        cn = max(0, cn + (mu - val - k))
        flag = (cp > h) or (cn > h)
        score = max(cp, cn)
        flags.append(flag)
        scores.append(score)
        mu_vals.append(mu)
        if flag:
            cp, cn = 0.0, 0.0  # reset on detection

    return pd.Series(flags, index=series.index), \
           pd.Series(scores, index=series.index), \
           pd.Series(mu_vals, index=series.index)

# -------------------------------------------------------------
# RUN ALL DETECTORS
# -------------------------------------------------------------
print(f"\nRunning MA+SD detector (window={MASD_WINDOW})...")
df['masd_flag'], df['masd_score'], df['masd_mean'], df['masd_upper'], df['masd_lower'] = \
    detect_masd(df['distance_mm'], window=MASD_WINDOW)

print("Running RoC detector...")
df['roc_flag'], df['roc_score'], df['roc_val'] = \
    detect_roc(df['distance_mm'])

print("Running Adaptive CUSUM detector...")
df['cusum_flag'], df['cusum_score'], df['cusum_baseline'] = \
    detect_adaptive_cusum(df['distance_mm'])

# -------------------------------------------------------------
# CONFIDENCE SCORING
# -------------------------------------------------------------
def compute_confidence(masd_f, roc_f, cusum_f, ml_f=False):
    """
    Confidence score from 0.0 to 1.0 based on method agreement.
    Multi-method agreement is the strongest signal of a true anomaly
    (only 10/451 detections in the paper were shared -- those are gold).
    """
    n_agree = int(masd_f) + int(roc_f) + int(cusum_f)
    score   = n_agree / 3.0
    if ml_f:
        score = min(1.0, score + 0.20)

    if score < 0.25:   tier = 'log'
    elif score < 0.60: tier = 'dashboard'
    elif score < 0.90: tier = 'alert'
    else:              tier = 'critical'

    return round(score, 3), tier

df['n_methods']  = df['masd_flag'].astype(int) + df['roc_flag'].astype(int) + df['cusum_flag'].astype(int)
df['conf_score'] = df['n_methods'] / 3.0
df['conf_tier']  = pd.cut(df['conf_score'],
                          bins=[-0.01, 0.24, 0.59, 0.89, 1.01],
                          labels=['log','dashboard','alert','critical'])

# -------------------------------------------------------------
# ANOMALY CLASSIFICATION (Rule-Based)
# -------------------------------------------------------------
def classify_anomaly(row):
    """
    Rule-based classifier using signal signatures.
    Works with zero training data -- pure domain logic.
    Replace with Random Forest once 200+ labeled events available.
    """
    if not (row['masd_flag'] or row['roc_flag'] or row['cusum_flag']):
        return 'normal'

    roc   = row['roc_val']       if not pd.isna(row['roc_val'])   else 0
    std10 = row['roll_std_10']   if not pd.isna(row['roll_std_10']) else 0
    level = row['distance_mm']
    cusum = row['cusum_flag']
    masd  = row['masd_flag']
    roc_f = row['roc_flag']

    # Sensor freeze: std of recent readings near zero
    if std10 < 0.2:
        return 'sensor_freeze'

    # Single isolated spike that returns to normal -> sensor noise spike
    if masd and not roc_f and not cusum:
        return 'sensor_spike'

    # Tank very close to sensor (near-zero distance) -> overflow
    if level < 30:
        return 'overflow'

    # Sudden large drop -> theft or rapid drain
    if roc_f and roc < -2.0:
        return 'sudden_drain_theft'

    # Sustained positive RoC -> refill
    if roc_f and roc > 2.0:
        return 'refill_event'

    # CUSUM fires, small RoC -> slow ongoing leak
    if cusum and abs(roc) < 1.5:
        return 'slow_leak'

    return 'unclassified_anomaly'

print("Classifying anomalies...")
df['anomaly_class'] = df.apply(classify_anomaly, axis=1)

# Hybrid flag: anomaly if ANY method fires
df['hybrid_flag'] = df['masd_flag'] | df['roc_flag'] | df['cusum_flag']

# -------------------------------------------------------------
# RESULTS SUMMARY
# -------------------------------------------------------------
print("\n" + "=" * 65)
print("STATISTICAL DETECTION RESULTS")
print("=" * 65)
print(f"Total readings        : {len(df)}")
print(f"\nMA+SD anomalies       : {df['masd_flag'].sum()}")
print(f"RoC anomalies         : {df['roc_flag'].sum()}")
print(f"Adaptive CUSUM        : {df['cusum_flag'].sum()}")
print(f"Hybrid (any method)   : {df['hybrid_flag'].sum()}")

# Method overlap
both_masd_roc   = (df['masd_flag'] & df['roc_flag']).sum()
both_masd_cusum = (df['masd_flag'] & df['cusum_flag']).sum()
both_roc_cusum  = (df['roc_flag']  & df['cusum_flag']).sum()
all_three       = (df['masd_flag'] & df['roc_flag'] & df['cusum_flag']).sum()
print(f"\nMethod overlap:")
print(f"  MA+SD n RoC         : {both_masd_roc}")
print(f"  MA+SD n CUSUM       : {both_masd_cusum}")
print(f"  RoC   n CUSUM       : {both_roc_cusum}")
print(f"  All 3 agree (HIGH)  : {all_three}")

print(f"\nAlert tier breakdown:")
print(df['conf_tier'].value_counts().to_string())

print(f"\nAnomaly classes:")
print(df['anomaly_class'].value_counts().to_string())

# Per-day breakdown (for multi-day datasets)
total_days = (df.index.max() - df.index.min()).total_seconds() / 86400
if total_days > 1.5:
    print(f"\nPer-day anomaly breakdown:")
    print(f"  {'Date':<14s} {'Total':>6s} {'MA+SD':>6s} {'RoC':>6s} {'CUSUM':>6s} {'Hybrid':>7s}")
    print(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for date, grp in df.groupby(df.index.date):
        print(f"  {str(date):<14s} {len(grp):>6d} {grp['masd_flag'].sum():>6d} "
              f"{grp['roc_flag'].sum():>6d} {grp['cusum_flag'].sum():>6d} "
              f"{grp['hybrid_flag'].sum():>7d}")

# -------------------------------------------------------------
# SAVE THRESHOLDS (for Flask production use)
# -------------------------------------------------------------
thresholds = {
    "masd": {"window": MASD_WINDOW, "n_std": 2.5, "note": "Applied to detrended signal -- removes linear drain trend first"},
    "roc":  {"drop_threshold": -1.0, "rise_threshold": 1.0},
    "cusum": {"alpha": 0.15, "k": 0.5, "h": 10.0, "note": "alpha=0.15 tracks drain trend; h=10 avoids noise triggers"},
    "confidence_tiers": {
        "log": [0.0, 0.24],
        "dashboard": [0.25, 0.59],
        "alert": [0.60, 0.89],
        "critical": [0.90, 1.0]
    }
}
with open(THRESHOLDS_PATH, 'w') as f:
    json.dump(thresholds, f, indent=2)
print(f"\n[OK] Thresholds saved to {THRESHOLDS_PATH}")

df.to_csv(OUTPUT_PATH)
print(f"[OK] Results saved to {OUTPUT_PATH}")
print("\nReady for Step 3 (ML detectors)")
