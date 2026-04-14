"""
=============================================================================
STEP 2: STATISTICAL ANOMALY DETECTORS (Seasonally-Adjusted)
=============================================================================
Architecture:
  1. STL Decomposition extracts Trend + Seasonal + Residual
  2. Statistical detectors run on the RESIDUAL (not raw signal)
     - This ensures recurring daily/weekly patterns are NOT flagged
     - Only unexpected deviations (the "surprise") trigger anomalies

Detectors on residual:
  - MA+SD:  Flags residual spikes (point anomalies)
  - RoC:   Flags sudden jumps in residual (abrupt events)
  - CUSUM: Flags sustained residual drift (slow leaks)

Auto-scales parameters based on dataset size:
  - Small dataset  (< 2000 rows): window=50
  - Large dataset  (>= 2000 rows): window=100

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROCESSED_PATH = "data/processed.csv"
OUTPUT_PATH    = "data/statistical_results.csv"
THRESHOLDS_PATH= "models/stat_thresholds.json"

os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("output/plots", exist_ok=True)

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
# STL DECOMPOSITION (Signal Understanding -- runs FIRST)
# -------------------------------------------------------------
print("\n" + "=" * 65)
print("STL DECOMPOSITION -- Extracting Trend, Seasonal, Residual")
print("=" * 65)
print("""
Purpose:
  Signal = Trend + Seasonal + Residual
  - Trend:    Long-term drain slope (expected)
  - Seasonal: Recurring daily/weekly usage patterns (expected)
  - Residual: What's left = the "surprise" (anomalies live here)

  Statistical detectors will run on the RESIDUAL, not the raw signal.
  This prevents regular daily patterns from being flagged as anomalies.
""")

try:
    from statsmodels.tsa.seasonal import STL

    # Choose period based on data length
    stl_period = 60 if DATASET_SIZE > 120 else max(10, DATASET_SIZE // 5)
    print(f"STL period: {stl_period} (= {stl_period}-minute cycle)")
    print(f"Robust mode: True (downweights outliers during decomposition)")

    stl = STL(df['distance_mm'], period=stl_period, robust=True)
    result = stl.fit()

    df['stl_trend']    = result.trend
    df['stl_seasonal'] = result.seasonal
    df['stl_residual'] = result.resid

    residual_mean = df['stl_residual'].mean()
    residual_std  = df['stl_residual'].std()

    print(f"\nSTL Decomposition Results:")
    print(f"  Trend range:    {df['stl_trend'].min():.2f} -- {df['stl_trend'].max():.2f} mm")
    print(f"  Seasonal range: {df['stl_seasonal'].min():.2f} -- {df['stl_seasonal'].max():.2f} mm")
    print(f"  Residual mean:  {residual_mean:.4f} mm (should be ~0)")
    print(f"  Residual std:   {residual_std:.4f} mm")
    print(f"\n  Detectors will now run on the RESIDUAL component.")

    STL_SUCCESS = True

    # -- Save STL decomposition plot --
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('STL Decomposition -- Signal Understanding\n'
                 '(Separating expected patterns from anomalies)',
                 fontsize=14, fontweight='bold')

    axes[0].plot(df.index, df['distance_mm'], lw=0.5, color='#2563eb')
    axes[0].set_ylabel('Original (mm)')
    axes[0].set_title('(a) Original Signal')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, df['stl_trend'], lw=1, color='#059669')
    axes[1].set_ylabel('Trend (mm)')
    axes[1].set_title('(b) Trend Component (long-term drain direction)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df.index, df['stl_seasonal'], lw=0.5, color='#d97706')
    axes[2].set_ylabel('Seasonal (mm)')
    axes[2].set_title('(c) Seasonal Component (recurring daily/hourly patterns)')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(df.index, df['stl_residual'], lw=0.5, color='#6b7280')
    axes[3].axhline(y=3*residual_std, color='r', ls='--', alpha=0.7, label=f'+3σ ({3*residual_std:.1f}mm)')
    axes[3].axhline(y=-3*residual_std, color='r', ls='--', alpha=0.7, label=f'-3σ ({-3*residual_std:.1f}mm)')
    axes[3].set_ylabel('Residual (mm)')
    axes[3].set_title('(d) Residual = "Surprise" (anomaly detection runs HERE)')
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/plots/05_stl_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] output/plots/05_stl_decomposition.png")

except ImportError:
    print("[!] statsmodels not installed -- falling back to linear detrending")
    print("  Install: pip install statsmodels")
    from scipy import stats as scipy_stats
    x = np.arange(len(df))
    slope, intercept, _, _, _ = scipy_stats.linregress(x, df['distance_mm'].values)
    df['stl_trend']    = slope * x + intercept
    df['stl_seasonal'] = 0.0
    df['stl_residual'] = df['distance_mm'] - df['stl_trend']
    residual_std = df['stl_residual'].std()
    STL_SUCCESS = False

except Exception as e:
    print(f"[!] STL failed: {e} -- falling back to linear detrending")
    from scipy import stats as scipy_stats
    x = np.arange(len(df))
    slope, intercept, _, _, _ = scipy_stats.linregress(x, df['distance_mm'].values)
    df['stl_trend']    = slope * x + intercept
    df['stl_seasonal'] = 0.0
    df['stl_residual'] = df['distance_mm'] - df['stl_trend']
    residual_std = df['stl_residual'].std()
    STL_SUCCESS = False

# The signal that detectors will operate on:
detection_signal = df['stl_residual']
print(f"\n  Detection signal: STL residual (mean={detection_signal.mean():.4f}, std={detection_signal.std():.4f})")

# -------------------------------------------------------------
# DETECTOR 1: MA+SD on Residual
# -------------------------------------------------------------
def detect_masd(series, window=50, n_std=2.5):
    """
    Moving Average + Standard Deviation detector on STL residual.

    Since STL already removed trend and seasonality, the residual
    is approximately stationary. We detect point anomalies as
    residual values that deviate significantly from the local
    rolling statistics.
    """
    roll_mean = series.rolling(window, min_periods=window//2).mean()
    roll_std  = series.rolling(window, min_periods=window//2).std()

    deviation = (series - roll_mean).abs()
    flag  = deviation > n_std * roll_std
    score = (deviation / roll_std.replace(0, np.nan)).fillna(0)

    upper = roll_mean + n_std * roll_std
    lower = roll_mean - n_std * roll_std

    return flag, score, roll_mean, upper, lower

# -------------------------------------------------------------
# DETECTOR 2: Rate of Change on Residual
# -------------------------------------------------------------
def detect_roc(series, drop_threshold=-1.0, rise_threshold=1.0):
    """
    Flags sudden jumps in the residual.
    Since seasonal patterns are removed, a sudden residual change
    means something unexpected happened (theft, valve failure, etc.)
    Regular daily drains are in the seasonal component -- invisible here.
    """
    roc = series.diff(1)
    flag  = (roc < drop_threshold) | (roc > rise_threshold)
    score = roc.abs() / abs(drop_threshold)
    return flag, score, roc

# -------------------------------------------------------------
# DETECTOR 3: Adaptive CUSUM on Residual
# -------------------------------------------------------------
def detect_adaptive_cusum(series, alpha=0.15, k=0.5, h=10.0):
    """
    CUSUM with EWMA adaptive baseline on the residual.
    The residual should be mean-near-zero if STL is working well.
    CUSUM accumulates sustained deviations -- detecting slow leaks
    that are NOT part of the normal seasonal pattern.
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
# RUN ALL DETECTORS ON RESIDUAL
# -------------------------------------------------------------
print(f"\n{'=' * 65}")
print("RUNNING DETECTORS ON STL RESIDUAL")
print("=" * 65)

print(f"\nRunning MA+SD on residual (window={MASD_WINDOW})...")
df['masd_flag'], df['masd_score'], df['masd_mean'], df['masd_upper'], df['masd_lower'] = \
    detect_masd(detection_signal, window=MASD_WINDOW)

print("Running RoC on residual...")
df['roc_flag'], df['roc_score'], df['roc_val'] = \
    detect_roc(detection_signal)

print("Running Adaptive CUSUM on residual...")
df['cusum_flag'], df['cusum_score'], df['cusum_baseline'] = \
    detect_adaptive_cusum(detection_signal)

# Also store CUSUM pos/neg for plotting
mu = detection_signal.iloc[0]
cp_vals, cn_vals = [], []
cp, cn = 0.0, 0.0
alpha_c = 0.15; k_c = 0.5; h_c = 10.0
for val in detection_signal:
    mu = alpha_c * val + (1 - alpha_c) * mu
    cp = max(0, cp + (val - mu - k_c))
    cn = max(0, cn + (mu - val - k_c))
    cp_vals.append(cp)
    cn_vals.append(cn)
    if cp > h_c or cn > h_c:
        cp, cn = 0.0, 0.0
df['cusum_pos'] = cp_vals
df['cusum_neg'] = cn_vals

# -------------------------------------------------------------
# CONFIDENCE SCORING (3-method)
# -------------------------------------------------------------
df['n_methods'] = df['masd_flag'].astype(int) + df['roc_flag'].astype(int) + df['cusum_flag'].astype(int)
df['conf_score'] = df['n_methods'] / 3.0
df['conf_tier']  = pd.cut(df['conf_score'],
                          bins=[-0.01, 0.24, 0.59, 0.89, 1.01],
                          labels=['log','dashboard','alert','critical'])

# -------------------------------------------------------------
# ANOMALY CLASSIFICATION (Rule-Based)
# -------------------------------------------------------------
def classify_anomaly(row):
    """
    Rule-based classifier using residual signal signatures.
    Now operates on residually-adjusted features.
    """
    if not (row['masd_flag'] or row['roc_flag'] or row['cusum_flag']):
        return 'normal'

    roc   = row['roc_val']       if not pd.isna(row['roc_val'])   else 0
    std10 = row['roll_std_10']   if 'roll_std_10' in row.index and not pd.isna(row.get('roll_std_10', np.nan)) else 0
    level = row['distance_mm']
    cusum = row['cusum_flag']
    masd  = row['masd_flag']
    roc_f = row['roc_flag']

    # Sensor freeze: std of recent readings near zero
    if std10 < 0.2:
        return 'sensor_freeze'

    # Single isolated spike in residual -> sensor noise
    if masd and not roc_f and not cusum:
        return 'sensor_spike'

    # Tank very close to sensor -> overflow
    if level < 30:
        return 'overflow'

    # Sudden large residual drop -> theft or rapid drain
    if roc_f and roc < -2.0:
        return 'sudden_drain_theft'

    # Sustained positive RoC in residual -> unexpected refill
    if roc_f and roc > 2.0:
        return 'refill_event'

    # CUSUM fires on residual, small RoC -> slow leak
    if cusum and abs(roc) < 1.5:
        return 'slow_leak'

    return 'unclassified_anomaly'

print("Classifying anomalies...")
df['anomaly_class'] = df.apply(classify_anomaly, axis=1)

# Hybrid flag: anomaly if ANY statistical method fires on residual
df['hybrid_flag'] = df['masd_flag'] | df['roc_flag'] | df['cusum_flag']

# -------------------------------------------------------------
# RESULTS SUMMARY
# -------------------------------------------------------------
print("\n" + "=" * 65)
print("STATISTICAL DETECTION RESULTS (on STL Residual)")
print("=" * 65)
print(f"Total readings        : {len(df)}")
print(f"STL decomposed        : {'Yes' if STL_SUCCESS else 'Fallback (linear detrend)'}")
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
print(f"  MA+SD & RoC         : {both_masd_roc}")
print(f"  MA+SD & CUSUM       : {both_masd_cusum}")
print(f"  RoC   & CUSUM       : {both_roc_cusum}")
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
# PLOTS
# -------------------------------------------------------------
COLORS = {
    'signal': '#2E75B6', 'normal': '#70AD47', 'masd': '#FF6B35',
    'roc': '#E63946', 'cusum': '#9B2335', 'residual': '#6b7280',
    'band': '#D6E4F0',
}

# -- Plot: MA+SD on Residual --
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
fig.suptitle('Technique 1: MA+SD on STL Residual\n'
             '(Seasonal patterns removed before detection)', fontweight='bold')

ax1.plot(df.index, df['distance_mm'], color=COLORS['signal'], lw=0.5, alpha=0.7, label='Original signal')
anom = df[df['masd_flag']]
ax1.scatter(anom.index, anom['distance_mm'], color=COLORS['masd'], s=25, zorder=5,
            label=f'MA+SD anomalies ({len(anom)})', marker='x')
ax1.set_ylabel('Distance (mm)'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
ax1.set_title('(a) Anomalies mapped back to original signal')

ax2.plot(df.index, df['stl_residual'], color=COLORS['residual'], lw=0.5, alpha=0.7, label='STL Residual')
ax2.fill_between(df.index, df['masd_lower'], df['masd_upper'],
                 alpha=0.2, color=COLORS['band'], label='MA±2.5σ band')
ax2.scatter(anom.index, anom['stl_residual'], color=COLORS['masd'], s=25, zorder=5,
            label=f'Anomalies ({len(anom)})', marker='x')
ax2.set_ylabel('Residual (mm)'); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
ax2.set_title('(b) Detection on STL residual (trend + seasonal removed)')

plt.tight_layout()
plt.savefig('output/plots/02_masd.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] output/plots/02_masd.png")

# -- Plot: Adaptive CUSUM on Residual --
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
fig.suptitle('Technique 3: Adaptive CUSUM on STL Residual', fontweight='bold')

ax1.plot(df.index, df['stl_residual'], color=COLORS['residual'], lw=0.5, alpha=0.7, label='STL Residual')
ax1.plot(df.index, df['cusum_baseline'], color='orange', lw=1, ls='--',
         label='EWMA baseline (α=0.15)')
anom_c = df[df['cusum_flag']]
ax1.scatter(anom_c.index, anom_c['stl_residual'], color=COLORS['cusum'], s=30, zorder=5,
            label=f'CUSUM anomalies ({len(anom_c)})', marker='D')
ax1.set_ylabel('Residual (mm)'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
ax1.set_title('(a) CUSUM detection on residual')

ax2.plot(df.index, df['cusum_pos'], color='#C00000', lw=1, label='CUSUM+ (rising drift)')
ax2.plot(df.index, df['cusum_neg'], color='#0070C0', lw=1, label='CUSUM- (falling drift)')
ax2.axhline(y=10.0, color='black', ls='--', lw=1.5, label='Threshold h=10.0')
ax2.set_ylabel('CUSUM Statistic'); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
ax2.set_title('(b) CUSUM accumulator vs threshold')

plt.tight_layout()
plt.savefig('output/plots/03_adaptive_cusum.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] output/plots/03_adaptive_cusum.png")

# -- Plot: All 3 methods comparison --
fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
fig.suptitle('All 3 Statistical Methods on STL Residual', fontsize=14, fontweight='bold')

# Show residual with each method's detections
methods_plot = [
    ('masd_flag',  'MA+SD',         COLORS['masd']),
    ('roc_flag',   'Rate of Change', COLORS['roc']),
    ('cusum_flag', 'Adaptive CUSUM', COLORS['cusum']),
]
for i, (col, name, color) in enumerate(methods_plot):
    ax = axes[i]
    ax.plot(df.index, df['stl_residual'], color=COLORS['residual'], lw=0.5, alpha=0.6)
    anom = df[df[col]]
    ax.scatter(anom.index, anom['stl_residual'], color=color, s=15, zorder=5,
               label=f'{name} -- {len(anom)} detections')
    ax.set_ylabel('Residual (mm)', fontsize=8)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

# Bottom: original signal with hybrid
axes[3].plot(df.index, df['distance_mm'], color=COLORS['signal'], lw=0.5, alpha=0.6)
hybrid_pts = df[df['hybrid_flag']]
axes[3].scatter(hybrid_pts.index, hybrid_pts['distance_mm'], color='#DC2626', s=15, zorder=5,
                label=f'Hybrid (any method) -- {len(hybrid_pts)} detections')
axes[3].set_ylabel('Distance (mm)', fontsize=8)
axes[3].legend(loc='upper left', fontsize=9)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/plots/04_all_methods_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] output/plots/04_all_methods_comparison.png")

# -------------------------------------------------------------
# SAVE THRESHOLDS
# -------------------------------------------------------------
thresholds = {
    "masd": {"window": MASD_WINDOW, "n_std": 2.5, "note": "Applied to STL residual -- seasonal patterns removed first"},
    "roc":  {"drop_threshold": -1.0, "rise_threshold": 1.0, "note": "Applied to STL residual differences"},
    "cusum": {"alpha": 0.15, "k": 0.5, "h": 10.0, "note": "Applied to STL residual -- tracks unexpected drift only"},
    "stl": {"period": 60 if DATASET_SIZE > 120 else max(10, DATASET_SIZE // 5), "robust": True},
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
print("\nReady for Step 3 (Prophet signal understanding)")
