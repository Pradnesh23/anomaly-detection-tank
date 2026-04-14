"""
=============================================================================
STEP 3: TIME SERIES SIGNAL UNDERSTANDING -- Prophet Forecasting
=============================================================================
Prophet provides CONTEXTUAL UNDERSTANDING, not anomaly detection:
  - Learns daily/weekly seasonal patterns
  - Forecasts expected future behavior
  - Provides context: "Is this normal for this time of day/week?"

STL decomposition has already been done in Step 2 (02_statistical_detectors.py).
This step adds Prophet's learned patterns without generating detection flags.

Combined 3-method confidence score is computed from statistical detectors only:
  MA+SD (0.33) + RoC (0.33) + CUSUM (0.33)

Reads:  data/statistical_results.csv
Writes: data/tsa_results.csv
        output/plots/06_prophet_forecast.png
        models/prophet_seasonal.json
=============================================================================
"""

import pandas as pd
import numpy as np
import os, json, warnings
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
# PROPHET -- Signal Understanding (NOT Detection)
# -------------------------------------------------------------
print("\n" + "=" * 65)
print("PROPHET FORECASTING -- Signal Understanding")
print("=" * 65)
print("""
Purpose:
  Prophet models daily/weekly usage patterns to provide CONTEXT:
  - "What level is expected at this time of day?"
  - "Is this drain rate normal for a Tuesday afternoon?"
  - Forecast future levels for predictive alerts

  Prophet does NOT generate anomaly detection flags.
  Detection is handled by the 3 statistical methods on STL residuals.
""")

PROPHET_SUCCESS = False
try:
    from prophet import Prophet
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Prepare Prophet input format
    df_prophet = pd.DataFrame({
        'ds': df.index,
        'y':  df['distance_mm'].values
    })

    # Subsample if too large (Prophet is slow on >10K rows)
    if len(df_prophet) > 10000:
        df_prophet_train = df_prophet.iloc[::5].reset_index(drop=True)
        print(f"  Subsampled for training: {len(df_prophet_train)} rows "
              f"(from {len(df_prophet)})")
    else:
        df_prophet_train = df_prophet.copy()

    # Configure Prophet
    enable_weekly = total_days >= 3
    print(f"  daily_seasonality:  True")
    print(f"  weekly_seasonality: {enable_weekly} (data spans {total_days:.1f} days)")
    print(f"  changepoint_prior_scale: 0.05 (low = stable baseline)")
    print(f"  interval_width: 0.95 (95% CI)")

    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=enable_weekly,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.95
    )
    m.fit(df_prophet_train)

    # Predict on all timestamps
    future = pd.DataFrame({'ds': df.index})
    forecast = m.predict(future)

    # Store Prophet understanding (NOT as detection flags)
    df['prophet_yhat']       = forecast['yhat'].values
    df['prophet_yhat_lower'] = forecast['yhat_lower'].values
    df['prophet_yhat_upper'] = forecast['yhat_upper'].values
    df['prophet_deviation']  = (df['distance_mm'] - df['prophet_yhat']).abs()

    # Context: is the current reading unusual for this time of day?
    ci_width = df['prophet_yhat_upper'] - df['prophet_yhat_lower']
    df['prophet_surprise'] = (df['prophet_deviation'] / (ci_width + 1e-6)).clip(0, 5)

    # NOT generating prophet_flag -- Prophet provides context, not detection
    # Legacy compatibility: set prophet_flag to False
    df['prophet_flag'] = False
    df['prophet_score'] = df['prophet_surprise']

    prophet_outside_ci = ((df['distance_mm'] < df['prophet_yhat_lower']) |
                          (df['distance_mm'] > df['prophet_yhat_upper'])).sum()

    print(f"\nProphet Understanding:")
    print(f"  Mean prediction: {df['prophet_yhat'].mean():.2f} mm")
    print(f"  CI width (avg):  {ci_width.mean():.2f} mm")
    print(f"  Readings outside 95% CI: {prophet_outside_ci} (contextual surprise)")
    print(f"  NOTE: These are NOT counted as anomaly detections.")
    print(f"        Prophet provides understanding, not flags.")

    # Save Prophet seasonal data
    try:
        hours = np.arange(24)
        hour_effects = []
        for h in hours:
            mask = forecast['ds'].dt.hour == h
            if mask.any():
                effect = forecast.loc[mask, 'yhat'].mean() - forecast['yhat'].mean()
                hour_effects.append(float(effect))
            else:
                hour_effects.append(0.0)

        prophet_export = {
            "daily_seasonality": hour_effects,
            "trend_slope": float(forecast['trend'].diff().mean()),
            "trend_intercept": float(forecast['trend'].iloc[0]),
            "ci_width_mean": float(ci_width.mean()),
            "training_rows": len(df_prophet_train),
            "weekly_seasonality_enabled": enable_weekly,
            "note": "Prophet provides signal understanding, not anomaly detection flags"
        }
        with open("models/prophet_seasonal.json", "w") as f:
            json.dump(prophet_export, f, indent=2)
        print("[OK] Saved: models/prophet_seasonal.json")
    except Exception:
        pass

    # Save forecast plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(df.index, df['prophet_yhat_lower'], df['prophet_yhat_upper'],
                    alpha=0.2, color='#2563eb', label='95% CI')
    ax.plot(df.index, df['distance_mm'], linewidth=0.5, color='#1a1a2e',
            alpha=0.7, label='Actual')
    ax.plot(df.index, df['prophet_yhat'], linewidth=1, color='#2563eb',
            label='Prophet Forecast')
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance (mm)')
    ax.set_title('Prophet Forecast -- Signal Understanding\n'
                 '(Models expected behavior at each time of day)')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig('output/plots/06_prophet_forecast.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/plots/06_prophet_forecast.png")

    PROPHET_SUCCESS = True

except ImportError as e:
    print(f"[!] Prophet skipped (not installed): {e}")
    print("  Install: pip install prophet")
    df['prophet_flag']       = False
    df['prophet_score']      = 0.0
    df['prophet_yhat']       = df['distance_mm']
    df['prophet_yhat_lower'] = df['distance_mm']
    df['prophet_yhat_upper'] = df['distance_mm']
    df['prophet_surprise']   = 0.0

except Exception as e:
    print(f"[!] Prophet failed: {e}")
    df['prophet_flag']       = False
    df['prophet_score']      = 0.0
    df['prophet_yhat']       = df['distance_mm']
    df['prophet_yhat_lower'] = df['distance_mm']
    df['prophet_yhat_upper'] = df['distance_mm']
    df['prophet_surprise']   = 0.0

# Also keep stl_flag=False for compatibility (STL is preprocessing, not detection)
df['stl_flag'] = False
df['stl_score'] = 0.0

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
print(f"Signal understanding: STL decomposition + Prophet forecasting")

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
