"""
=============================================================================
RUN_ALL.py -- Execute the complete pipeline end to end
=============================================================================
Run this from the tank_monitor/ directory:
    python run_all.py

Or run individual steps:
    python ml_pipeline/00_anomaly_definitions.py
    python ml_pipeline/01_preprocessing.py
    python ml_pipeline/02_statistical_detectors.py
    ...

Pipeline:
  Step 0: Anomaly definitions + threshold rationale
  Step 1: Data loading, cleaning, gap filling, feature engineering
  Step 2: Statistical detectors (MA+SD, RoC, Adaptive CUSUM)
  Step 3: Time Series Analysis (STL Decomposition + Prophet)
  Step 4: Evaluation (metrics, plots, SOTA comparison)
  Step 5: Synthetic data generation
  Step 6: TinyML export (C headers for ESP32)
=============================================================================
"""
import subprocess, sys, os, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STEPS = [
    ("ml_pipeline/00_anomaly_definitions.py",   "Formal anomaly taxonomy, threshold justification"),
    ("ml_pipeline/01_preprocessing.py",         "Data loading, cleaning, gap filling, feature engineering"),
    ("ml_pipeline/02_statistical_detectors.py", "MA+SD, RoC, Adaptive CUSUM, confidence scoring"),
    ("ml_pipeline/03_ml_detectors.py",          "STL Decomposition + Prophet (Time Series Analysis)"),
    ("ml_pipeline/05_evaluation.py",            "Metrics, plots, SOTA comparison, deployment benchmarks"),
    ("ml_pipeline/07_datasets_and_synthetic.py","External dataset guide + synthetic data generation"),
    ("ml_pipeline/08_tinyml_export.py",         "TinyML C headers for ESP32 + deployment analysis"),
]

print("=" * 65)
print("TANK MONITOR -- COMPLETE ANOMALY DETECTION PIPELINE")
print("=" * 65)
print("\nDetection methods:")
print("  Statistical : MA+SD (detrended) | Rate of Change | Adaptive CUSUM")
print("  TSA         : STL Decomposition | Prophet Forecasting")
print("  TinyML      : C export for ESP32 edge deployment")
print("\nInstalling required packages...")

packages = ["pandas", "numpy", "scikit-learn", "matplotlib", "flask",
            "joblib", "statsmodels", "scipy"]
for pkg in packages:
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=False)

print("\nOptional packages (install manually if needed):")
print("  pip install prophet      # for Prophet forecasting")
print("  pip install flask        # for web dashboard")
print()

results = {}
for i, (script, desc) in enumerate(STEPS):
    print(f"\n{'-'*65}")
    print(f"STEP {i}: {desc}")
    print(f"{'-'*65}")

    if not os.path.exists(script):
        results[script] = "[!]  Script not found"
        print(f"  [!] {script} not found -- skipping")
        continue

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            timeout=600   # 10 min timeout for Prophet
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            results[script] = f"[OK]  {elapsed:.1f}s"
        else:
            results[script] = f"[!]  Failed (returncode {result.returncode})"
    except subprocess.TimeoutExpired:
        results[script] = "[!]  Timeout (>600s)"
    except Exception as e:
        results[script] = f"[!]  Error: {e}"

print(f"\n{'='*65}")
print("PIPELINE SUMMARY")
print("="*65)
for script, status in results.items():
    print(f"  {status}  {script}")

print(f"\nOutputs:")
print(f"  data/processed.csv             -- cleaned, feature-engineered dataset")
print(f"  data/statistical_results.csv   -- MA+SD, RoC, CUSUM flags + scores")
print(f"  data/tsa_results.csv           -- STL + Prophet flags + final 5-method scores")
print(f"  data/synthetic_labeled.csv     -- labeled synthetic anomaly data")
print(f"  models/stat_thresholds.json    -- calibrated thresholds")
print(f"  models/tinyml_detectors.h      -- C code for ESP32 (TinyML)")
print(f"  models/prophet_seasonal.h      -- Prophet seasonality for ESP32")
print(f"  models/prophet_seasonal.json   -- Prophet learned patterns")
print(f"  output/plots/                  -- all figures for the paper")
print(f"  output/evaluation_report.txt   -- Precision/Recall/F1/MCC table")
print(f"  output/anomaly_taxonomy.txt    -- Formal anomaly definitions")
print(f"  output/parameter_rationale.txt -- Threshold justification")
print(f"  output/dataset_report.txt      -- Sensor data characteristics")
print(f"  output/deployment_analysis.txt -- TinyML deployment benchmarks")
print(f"\nTo start the web dashboard:")
print(f"  python 06_flask_backend.py")
print(f"  Open: http://localhost:5000/dashboard")
