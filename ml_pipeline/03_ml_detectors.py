"""
=============================================================================
STEP 3: TIME SERIES SIGNAL UNDERSTANDING -- STL Decomposition
=============================================================================
STL Decomposition provides CONTEXTUAL UNDERSTANDING of the signal:
  - Decomposes into Trend, Seasonal, and Residual components
  - Detectors in Step 2 already ran on STL residual
  - This step enriches the output and finalises the 3-method confidence score

Combined 3-method confidence score is computed from statistical detectors only:
  MA+SD (0.33) + RoC (0.33) + Adaptive CUSUM (0.33)

Reads:  data/statistical_results.csv
Writes: data/tsa_results.csv
=============================================================================
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

INPUT_PATH  = "data/statistical_results.csv"
OUTPUT_PATH = "data/tsa_results.csv"

os.makedirs("output/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------------------------------------------------
# LOAD
# -------------------------------------------------------------
df = pd.read_csv(INPUT_PATH, index_col='timestamp', parse_dates=True)
print(f"Loaded statistical results: {df.shape}")
print(f"Time range: {df.index.min()} -> {df.index.max()}")

total_hours = (df.index.max() - df.index.min()).total_seconds() / 3600
total_days  = total_hours / 24
print(f"Duration: {total_hours:.1f} hours ({total_days:.1f} days)")

# Confirm STL was done in step 2
if 'stl_residual' in df.columns:
    print(f"\n[OK] STL decomposition found from Step 2")
    print(f"  Residual std: {df['stl_residual'].std():.4f} mm")
else:
    print("[!] STL residual not found -- detectors may have run on raw signal")

print(f"\nStatistical detection flags from Step 2:")
print(f"  MA+SD:  {df['masd_flag'].sum()} anomalies")
print(f"  RoC:    {df['roc_flag'].sum()} anomalies")
print(f"  CUSUM:  {df['cusum_flag'].sum()} anomalies")

# -------------------------------------------------------------
# STL CONTEXT -- enrich output (decomposition already done in Step 2)
# -------------------------------------------------------------
print("\n" + "=" * 65)
print("STL DECOMPOSITION -- Signal Context")
print("=" * 65)
print("""
Purpose:
  STL breaks the signal into:
  - Trend: long-term direction (is the tank filling or draining overall?)
  - Seasonal: recurring daily/hourly cycle (expected fill/drain pattern)
  - Residual: noise after removing trend & season — detectors run on THIS

  Detection is performed on the residual, not the raw signal.
  This eliminates false alarms caused by expected daily drain cycles.
""")

if 'stl_trend' in df.columns and 'stl_seasonal' in df.columns:
    trend_range   = df['stl_trend'].max() - df['stl_trend'].min()
    season_amp    = df['stl_seasonal'].std()
    residual_std  = df['stl_residual'].std()
    print(f"  Trend range  : {trend_range:.2f} mm")
    print(f"  Seasonal amp : {season_amp:.2f} mm (std)")
    print(f"  Residual std : {residual_std:.2f} mm (detection target)")
else:
    print("  [!] STL columns not in dataset -- skipping context summary")

# Compatibility columns (no Prophet in this project)
df['prophet_flag']       = False
df['prophet_score']      = 0.0
df['stl_flag']           = False
df['stl_score']          = 0.0

# -------------------------------------------------------------
# FINAL 3-METHOD CONFIDENCE SCORE
# -------------------------------------------------------------
print("\n" + "=" * 65)
print("FINAL 3-METHOD CONFIDENCE SCORE")
print("=" * 65)

# Only statistical detectors count for detection
df['n_methods_total'] = (
    df['masd_flag'].astype(int) +
    df['roc_flag'].astype(int) +
    df['cusum_flag'].astype(int)
)

n_active = 3

df['conf_score_final'] = df['n_methods_total'] / n_active
df['conf_score_final'] = df['conf_score_final'].clip(0, 1)

# Assign tiers
df['conf_tier_final'] = pd.cut(
    df['conf_score_final'],
    bins=[-0.01, 0.19, 0.39, 0.69, 1.01],
    labels=['log', 'dashboard', 'alert', 'critical']
)

# Final hybrid flag: ANY statistical method fires
df['hybrid_flag_final'] = df['n_methods_total'] >= 1

# Results
print(f"\nDetection method: 3 statistical detectors on STL residual")
print(f"Signal understanding: STL decomposition (Trend + Seasonal + Residual)")

print(f"\nPer-method anomaly counts:")
print(f"  MA+SD:    {df['masd_flag'].sum()}")
print(f"  RoC:      {df['roc_flag'].sum()}")
print(f"  CUSUM:    {df['cusum_flag'].sum()}")

print(f"\nMethod agreement distribution:")
for n in range(n_active + 1):
    count = (df['n_methods_total'] == n).sum()
    pct = count / len(df) * 100
    label = "normal" if n == 0 else f"{n}/{n_active} methods agree"
    print(f"  {n} methods: {count:>6d} ({pct:>5.1f}%) -- {label}")

print(f"\nConfidence tier breakdown:")
print(df['conf_tier_final'].value_counts().sort_index().to_string())

# Consensus labels (for evaluation)
df['consensus_anomaly'] = df['n_methods_total'] >= 2  # >= 2 of 3 methods

# -------------------------------------------------------------
# SAVE
# -------------------------------------------------------------
df.to_csv(OUTPUT_PATH)
print(f"\n[OK] Results saved to {OUTPUT_PATH}")
print(f"  Shape: {df.shape}")
print(f"  Total anomalies (any method):    {df['hybrid_flag_final'].sum()}")
print(f"  High-confidence (>=2 agree):     {df['consensus_anomaly'].sum()}")
print(f"\nReady for Step 5 (evaluation)")
