"""
=============================================================================
STEP 3: TIME SERIES ANALYSIS DETECTORS -- STL Decomposition + Prophet
=============================================================================
Replaces the original ML detectors (Isolation Forest) with time series
analysis methods that understand trend and seasonality:

  1. STL Decomposition -- Seasonal-Trend decomposition using Loess
     - Decomposes signal into Trend + Seasonal + Residual
     - Anomaly if |Residual| > 3sigma
     
  2. Prophet (Meta's forecasting library)
     - Learns daily/weekly seasonal patterns
     - Anomaly if reading outside 95% confidence interval

  3. Combined 5-method confidence score
     - MA+SD (0.2) + RoC (0.2) + CUSUM (0.2) + STL (0.2) + Prophet (0.2)

Reads:  data/statistical_results.csv
Writes: data/tsa_results.csv
        output/plots/05_stl_decomposition.png
        output/plots/06_prophet_forecast.png
        models/prophet_seasonal.json

Addresses reviewer feedback:
  - "comparison with recent intelligent anomaly detection" (R1 + R2)
  - "more comprehensive experiments" (R2)
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

# -------------------------------------------------------------
# DETECTOR 4: STL DECOMPOSITION
# -------------------------------------------------------------
print("\n" + "=" * 65)
print("DETECTOR 4: STL DECOMPOSITION (Seasonal-Trend via Loess)")
print("=" * 65)

try:
    from statsmodels.tsa.seasonal import STL
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Choose period based on data length
    # 60 = 1-hour cycle (for sub-day data)
    # For multi-day, we still use 60 (hourly seasonality)
    stl_period = 60 if len(df) > 120 else max(10, len(df) // 5)
    print(f"STL period: {stl_period} (= {stl_period} minute cycle)")
    print(f"Robust mode: True (downweights outliers during decomposition)")

    # Run STL decomposition
    stl = STL(df['distance_mm'], period=stl_period, robust=True)
    result = stl.fit()

    df['stl_trend']    = result.trend
    df['stl_seasonal'] = result.seasonal
    df['stl_residual'] = result.resid

    # Anomaly detection on residuals
    residual_mean = df['stl_residual'].mean()
    residual_std  = df['stl_residual'].std()
    stl_threshold = 3.0  # standard 3-sigma rule

    df['stl_flag']  = (df['stl_residual'].abs() > stl_threshold * residual_std)
    df['stl_score'] = (df['stl_residual'].abs() / residual_std).clip(0, 5)

    stl_anomalies = df['stl_flag'].sum()
    print(f"\nSTL Results:")
    print(f"  Residual mean: {residual_mean:.4f} mm")
    print(f"  Residual std:  {residual_std:.4f} mm")
    print(f"  Threshold:     +-{stl_threshold * residual_std:.4f} mm (3sigma)")
    print(f"  Anomalies:     {stl_anomalies}")

    # Save decomposition plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('STL Decomposition -- Tank Level Signal', fontsize=14, fontweight='bold')

    axes[0].plot(df.index, df['distance_mm'], linewidth=0.5, color='#2563eb')
    axes[0].set_ylabel('Original (mm)')
    axes[0].set_title('Original Signal')

    axes[1].plot(df.index, df['stl_trend'], linewidth=1, color='#059669')
    axes[1].set_ylabel('Trend (mm)')
    axes[1].set_title('Trend Component (long-term direction)')

    axes[2].plot(df.index, df['stl_seasonal'], linewidth=0.5, color='#d97706')
    axes[2].set_ylabel('Seasonal (mm)')
    axes[2].set_title('Seasonal Component (recurring pattern)')

    axes[3].plot(df.index, df['stl_residual'], linewidth=0.5, color='#6b7280')
    axes[3].axhline(y=stl_threshold * residual_std, color='r', linestyle='--',
                    alpha=0.7, label=f'+{stl_threshold}sigma')
    axes[3].axhline(y=-stl_threshold * residual_std, color='r', linestyle='--',
                    alpha=0.7, label=f'-{stl_threshold}sigma')

    # Mark anomalies
    anomaly_mask = df['stl_flag']
    if anomaly_mask.any():
        axes[3].scatter(df.index[anomaly_mask], df['stl_residual'][anomaly_mask],
                       color='red', s=12, zorder=5, label='Anomaly')
    axes[3].set_ylabel('Residual (mm)')
    axes[3].set_title('Residual Component (anomalies live here)')
    axes[3].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('output/plots/05_stl_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/plots/05_stl_decomposition.png")

    STL_SUCCESS = True

except ImportError as e:
    print(f"[!] STL skipped (statsmodels not available): {e}")
    print("  Install: pip install statsmodels")
    df['stl_flag']     = False
    df['stl_score']    = 0.0
    df['stl_trend']    = df['distance_mm']
    df['stl_seasonal'] = 0.0
    df['stl_residual'] = 0.0
    STL_SUCCESS = False

except Exception as e:
    print(f"[!] STL failed: {e}")
    df['stl_flag']     = False
    df['stl_score']    = 0.0
    df['stl_trend']    = df['distance_mm']
    df['stl_seasonal'] = 0.0
    df['stl_residual'] = 0.0
    STL_SUCCESS = False


# -------------------------------------------------------------
# DETECTOR 5: PROPHET
# -------------------------------------------------------------
print("\n" + "=" * 65)
print("DETECTOR 5: PROPHET (Meta's Forecasting Library)")
print("=" * 65)

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
        # Take every 5th point for training, predict on all
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

    # Map back to main dataframe
    df['prophet_yhat']       = forecast['yhat'].values
    df['prophet_yhat_lower'] = forecast['yhat_lower'].values
    df['prophet_yhat_upper'] = forecast['yhat_upper'].values

    # Anomaly: reading outside 95% CI
    df['prophet_flag'] = (
        (df['distance_mm'] < df['prophet_yhat_lower']) |
        (df['distance_mm'] > df['prophet_yhat_upper'])
    )
    df['prophet_score'] = (
        (df['distance_mm'] - df['prophet_yhat']).abs() /
        (df['prophet_yhat_upper'] - df['prophet_yhat_lower'] + 1e-6)
    ).clip(0, 5)

    prophet_anomalies = df['prophet_flag'].sum()
    print(f"\nProphet Results:")
    print(f"  Mean prediction: {df['prophet_yhat'].mean():.2f} mm")
    print(f"  CI width (avg):  {(df['prophet_yhat_upper'] - df['prophet_yhat_lower']).mean():.2f} mm")
    print(f"  Anomalies:       {prophet_anomalies}")

    # Save Prophet's learned daily seasonality for TinyML export
    try:
        daily_comp = m.plot_components(forecast)
        plt.close('all')

        # Extract hourly seasonal values
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
            "ci_width_mean": float((forecast['yhat_upper'] - forecast['yhat_lower']).mean()),
            "training_rows": len(df_prophet_train),
            "weekly_seasonality_enabled": enable_weekly
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

    # Mark anomalies
    anomaly_mask = df['prophet_flag']
    if anomaly_mask.any():
        ax.scatter(df.index[anomaly_mask], df['distance_mm'][anomaly_mask],
                  color='red', s=12, zorder=5, label=f'Anomaly ({prophet_anomalies})')

    ax.set_xlabel('Time')
    ax.set_ylabel('Distance (mm)')
    ax.set_title('Prophet Forecast -- Anomaly Detection via 95% Confidence Interval')
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
    PROPHET_SUCCESS = False

except Exception as e:
    print(f"[!] Prophet failed: {e}")
    df['prophet_flag']       = False
    df['prophet_score']      = 0.0
    df['prophet_yhat']       = df['distance_mm']
    df['prophet_yhat_lower'] = df['distance_mm']
    df['prophet_yhat_upper'] = df['distance_mm']
    PROPHET_SUCCESS = False


# -------------------------------------------------------------
# COMBINED 5-METHOD CONFIDENCE SCORE
# -------------------------------------------------------------
print("\n" + "=" * 65)
print("COMBINED 5-METHOD CONFIDENCE SCORE")
print("=" * 65)

# Equal-weight voting: each method contributes 0.2
df['n_methods_total'] = (
    df['masd_flag'].astype(int) +
    df['roc_flag'].astype(int) +
    df['cusum_flag'].astype(int) +
    df['stl_flag'].astype(int) +
    df['prophet_flag'].astype(int)
)

# Active methods count (for proper normalization)
n_active = 3  # statistical always active
if STL_SUCCESS:     n_active += 1
if PROPHET_SUCCESS: n_active += 1

df['conf_score_final'] = df['n_methods_total'] / n_active
df['conf_score_final'] = df['conf_score_final'].clip(0, 1)

# Re-assign tiers using the new 5-method score
df['conf_tier_final'] = pd.cut(
    df['conf_score_final'],
    bins=[-0.01, 0.19, 0.39, 0.69, 1.01],
    labels=['log', 'dashboard', 'alert', 'critical']
)

# Final hybrid flag
df['hybrid_flag_final'] = df['n_methods_total'] >= 1

# Results
print(f"\nActive methods: {n_active} "
      f"({'STL' if STL_SUCCESS else 'no STL'}, "
      f"{'Prophet' if PROPHET_SUCCESS else 'no Prophet'})")
print(f"\nPer-method anomaly counts:")
print(f"  MA+SD:    {df['masd_flag'].sum()}")
print(f"  RoC:      {df['roc_flag'].sum()}")
print(f"  CUSUM:    {df['cusum_flag'].sum()}")
print(f"  STL:      {df['stl_flag'].sum()}")
print(f"  Prophet:  {df['prophet_flag'].sum()}")
print(f"\nMethod agreement distribution:")
for n in range(n_active + 1):
    count = (df['n_methods_total'] == n).sum()
    pct = count / len(df) * 100
    label = "normal" if n == 0 else f"{n}/{n_active} methods agree"
    print(f"  {n} methods: {count:>6d} ({pct:>5.1f}%) -- {label}")

print(f"\nConfidence tier breakdown:")
print(df['conf_tier_final'].value_counts().sort_index().to_string())

# Consensus labels (for evaluation)
df['consensus_anomaly'] = df['n_methods_total'] >= 3  # >=3 methods agree

# -------------------------------------------------------------
# SAVE
# -------------------------------------------------------------
df.to_csv(OUTPUT_PATH)
print(f"\n[OK] Results saved to {OUTPUT_PATH}")
print(f"  Shape: {df.shape}")
print(f"  Total anomalies (any method):    {df['hybrid_flag_final'].sum()}")
print(f"  High-confidence (>=3 agree):      {df['consensus_anomaly'].sum()}")
print(f"\nReady for Step 5 (evaluation)")
