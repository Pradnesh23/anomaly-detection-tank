"""
=============================================================================
STEP 0: ANOMALY DEFINITIONS & METHODOLOGY DOCUMENTATION
=============================================================================
Formal anomaly taxonomy, threshold justification, and labeling strategy.

Addresses reviewer feedback:
  - "clearer details on anomaly definition" (R1 + R2)
  - "anomaly labeling" (R1)
  - "methodological clarity" (TPC)

This module is imported by other pipeline scripts for consistent definitions.
Run standalone to generate the methodology report.

Writes: output/anomaly_taxonomy.txt
        output/parameter_rationale.txt
=============================================================================
"""

import os, json

os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------------------------------------------------
# ANOMALY TAXONOMY
# -------------------------------------------------------------
# Three categories following Chandola et al. (2009) classification:
#   1. Point Anomaly     -- single reading deviates from population
#   2. Contextual Anomaly -- reading unusual given temporal context
#   3. Collective Anomaly -- sequence of readings form abnormal pattern

ANOMALY_TAXONOMY = {
    "point": {
        "description": "A single data point that deviates significantly from the "
                       "expected value, irrespective of temporal context.",
        "mathematical_definition": "|x_t - u_w| > k x sigma_w, where u_w and sigma_w are "
                                   "the mean and std of a local window w",
        "subtypes": {
            "sensor_spike": {
                "description": "Isolated reading far from local context due to "
                               "HC-SR04 ultrasonic echo noise or multipath interference",
                "detection_method": "MA+SD (primary)",
                "criteria": "Single reading > u_50 + 2.5sigma_50 on detrended signal, "
                           "not accompanied by RoC or CUSUM flags",
                "example": "One reading at 120mm when surrounding readings are ~42mm"
            },
        }
    },
    "contextual": {
        "description": "A data point that is anomalous within a specific temporal "
                       "context but might be normal in a different context.",
        "mathematical_definition": "x_t falls outside the expected seasonal/trend "
                                   "envelope: |x_t - f(t)| > k x sigma_residual, "
                                   "where f(t) is the STL/Prophet model prediction",
        "subtypes": {
            "sudden_drain_theft": {
                "description": "Rapid unexpected drop in liquid level, potentially "
                               "indicating theft, valve failure, or pipe burst",
                "detection_method": "RoC (primary) + MA+SD (secondary)",
                "criteria": "RoC < -2.0 mm/min for >=2 consecutive readings",
                "example": "Distance jumps from 42mm to 72mm in 2 minutes"
            },
            "overflow": {
                "description": "Tank approaches or exceeds capacity; distance sensor "
                               "reads near-zero (liquid surface very close to sensor)",
                "detection_method": "MA+SD + RoC + CUSUM (all three typically fire)",
                "criteria": "distance_mm < 30mm (less than 12% of tank height 250mm)",
                "example": "Readings drop to 15-25mm range, inlet valve stuck open"
            },
            "refill_event": {
                "description": "Rapid intentional filling of tank. May be normal "
                               "operation but is flagged for logging and pattern analysis",
                "detection_method": "RoC (primary)",
                "criteria": "RoC > +2.0 mm/min sustained for >=3 readings",
                "example": "Distance drops from 80mm to 40mm during scheduled refill"
            },
        }
    },
    "collective": {
        "description": "A sequence of data points that together constitute an anomaly, "
                       "even though individual points may appear normal.",
        "mathematical_definition": "CUSUM accumulator S+ or S- exceeds threshold h "
                                   "after sustained small deviations from baseline",
        "subtypes": {
            "slow_leak": {
                "description": "Gradual persistent drift in liquid level indicating "
                               "a small leak, pipe seepage, or evaporation anomaly",
                "detection_method": "Adaptive CUSUM (primary)",
                "criteria": "CUSUM accumulates deviations > h=10.0 over 30+ minutes "
                           "with individual RoC < 1.5 mm/min",
                "example": "Level drops 0.3mm/min for 40 minutes (12mm total, "
                           "missed by RoC since each step is below +-1.0 threshold)"
            },
            "sensor_freeze": {
                "description": "Sensor output stuck at constant value due to hardware "
                               "malfunction, loose wiring, or condensation on sensor face",
                "detection_method": "Rule-based (std of recent readings)",
                "criteria": "std(last 20 readings) < 0.2mm",
                "example": "20+ consecutive readings at exactly 41.0mm with no variation"
            },
        }
    }
}

# -------------------------------------------------------------
# THRESHOLD JUSTIFICATION TABLE
# -------------------------------------------------------------
# Every parameter with domain-specific rationale

PARAMETER_RATIONALE = {
    "MA+SD": {
        "window": {
            "value": 50,
            "unit": "readings (~50 minutes at 1 reading/min)",
            "rationale": "50 minutes captures a typical stable operating segment. "
                         "Shorter windows (10-20) are too sensitive to HC-SR04 noise "
                         "(+-1.12mm measured noise floor). Longer windows (100+) smooth "
                         "out real events like refills.",
            "sensitivity_analysis": "Tested window in {20, 30, 50, 75, 100}. "
                                    "Window=50 minimized false positives while detecting "
                                    "all injected anomalies in synthetic data."
        },
        "n_std": {
            "value": 2.5,
            "unit": "standard deviations",
            "rationale": "Applied on detrended signal (after removing linear drain slope). "
                         "2.5sigma on detrended residuals corresponds to ~3mm deviation, "
                         "which is at the HC-SR04 accuracy limit. Values below 2.0sigma "
                         "flag normal sensor jitter. Values above 3.0sigma miss subtle "
                         "anomalies.",
            "note": "Original paper used 1.5sigma on raw signal -> caused >80% false "
                    "positive rate due to unaccounted drain trend. Detrending + 2.5sigma "
                    "reduces FPR to <5%."
        },
        "detrending": {
            "value": "linear regression (scipy.stats.linregress)",
            "rationale": "Tank drain follows approximately linear trend within each "
                         "operating segment. Detrending separates expected drift "
                         "(normal) from unexpected deviation (anomaly)."
        }
    },
    "RoC": {
        "drop_threshold": {
            "value": -1.0,
            "unit": "mm/min",
            "rationale": "HC-SR04 sensor noise floor measured at +-1.12mm (computed as "
                         "std of consecutive differences during stable periods). "
                         "Threshold at -1.0 is at the noise boundary -- any drop "
                         "larger than this is likely a real physical event, not noise.",
            "physical_meaning": "1mm/min drain = 60mm/hour. Normal drain observed "
                                "at ~0.14mm/min (8.24mm/hour). A reading crossing "
                                "-1.0 represents 7x the normal drain rate."
        },
        "rise_threshold": {
            "value": 1.0,
            "unit": "mm/min",
            "rationale": "Symmetric to drop threshold. A rise of >1mm/min indicates "
                         "active refilling or sensor disturbance."
        }
    },
    "Adaptive_CUSUM": {
        "alpha": {
            "value": 0.15,
            "unit": "EWMA smoothing factor (0-1)",
            "rationale": "Controls how quickly the adaptive baseline tracks the signal. "
                         "alpha=0.15 gives effective memory of ~6 readings (1/(1-alpha)~1.18, "
                         "but in practice ~6-7 readings contribute significantly). "
                         "Fast enough to track the normal drain trend, slow enough "
                         "to not absorb anomalous spikes into the baseline.",
            "note": "Original alpha=0.03 was too slow -- CUSUM accumulated the normal "
                    "drain as 'drift' and fired continuously. alpha=0.15 fixed this."
        },
        "k": {
            "value": 0.5,
            "unit": "mm (allowance/slack parameter)",
            "rationale": "Allows +-0.5mm of natural variation before CUSUM starts "
                         "accumulating. This is approximately half the sensor noise "
                         "floor (1.12mm), providing tolerance for sensor jitter "
                         "without being so large that small leaks are ignored."
        },
        "h": {
            "value": 10.0,
            "unit": "mm (detection threshold)",
            "rationale": "CUSUM fires when accumulated deviation exceeds 10mm. "
                         "Given k=0.5 and typical noise, a genuine slow leak of "
                         "0.3mm/min above expected would take ~25-30 minutes to "
                         "accumulate to h=10 -- a reasonable detection delay for "
                         "this application.",
            "note": "Original h=5.0 triggered too frequently on normal sensor "
                    "fluctuations. h=10.0 reduces noise triggers while still "
                    "catching leaks within 30 minutes."
        }
    },
    "STL_Decomposition": {
        "period": {
            "value": 60,
            "unit": "readings (= 60 minutes = 1 hour cycle)",
            "rationale": "Primary seasonal cycle in tank usage is hourly (pump "
                         "cycles, usage patterns). For multi-day data, daily "
                         "seasonality (period=1440) is also relevant but requires "
                         ">=2 full days of data."
        },
        "robust": {
            "value": True,
            "rationale": "Robust STL uses iteratively reweighted least squares, "
                         "which downweights outliers during decomposition. Essential "
                         "for our data since anomalies exist in the signal -- without "
                         "robust mode, anomalies distort the seasonal component."
        },
        "residual_threshold": {
            "value": 3.0,
            "unit": "standard deviations of residual",
            "rationale": "After decomposing signal into Trend + Seasonal + Residual, "
                         "residuals exceeding 3sigma are flagged. This follows the standard "
                         "3-sigma rule (99.7% of normal readings within +-3sigma)."
        }
    },
    "Prophet": {
        "interval_width": {
            "value": 0.95,
            "unit": "confidence interval (probability)",
            "rationale": "95% CI means only 5% of normal readings should fall outside "
                         "the predicted bounds. Readings outside this band are "
                         "anomalous with high confidence."
        },
        "changepoint_prior_scale": {
            "value": 0.05,
            "unit": "regularization strength",
            "rationale": "Low value = stable baseline, prevents Prophet from "
                         "overfitting to the gradual drain trend. Higher values "
                         "(0.5+) would cause Prophet to absorb anomalies as "
                         "changepoints, missing them during detection."
        },
        "daily_seasonality": {
            "value": True,
            "rationale": "Models the daily fill/drain cycle pattern. Essential for "
                         "detecting contextual anomalies (e.g., unexpected drain "
                         "during typically stable hours)."
        },
        "weekly_seasonality": {
            "value": "True if data spans >=3 days, else False",
            "rationale": "Weekly patterns (weekday vs weekend usage) only meaningful "
                         "with multi-day data. Automatically enabled for 21K dataset "
                         "(5 days), disabled for 998-row dataset (16 hours)."
        }
    },
    "Confidence_Scoring": {
        "method": {
            "value": "Equal-weight voting: 0.2 x each of 5 detectors",
            "rationale": "Each detector contributes 0.2 to the final score (max=1.0). "
                         "Equal weighting is used because we lack sufficient labeled "
                         "data to learn optimal weights. Statistical and TSA methods "
                         "are complementary (statistical catches sudden events, TSA "
                         "catches contextual anomalies), so equal weighting is fair."
        },
        "tiers": {
            "log":       {"range": "0.00 - 0.19", "action": "Log only, no alert"},
            "dashboard": {"range": "0.20 - 0.39", "action": "Show on dashboard, no push alert"},
            "alert":     {"range": "0.40 - 0.69", "action": "Generate alert notification"},
            "critical":  {"range": "0.70 - 1.00", "action": "Critical alert + potential SMS/email"},
        }
    }
}

# -------------------------------------------------------------
# LABELING STRATEGY
# -------------------------------------------------------------

LABELING_STRATEGY = {
    "method": "Ensemble Consensus Pseudo-Labeling",
    "description": (
        "In the absence of manually labeled ground truth (common in real IoT "
        "deployments where labeled anomaly data is scarce), we employ ensemble "
        "consensus labeling as a proxy for ground truth."
    ),
    "rules": {
        "anomaly":   "Flagged by >=3 out of 5 detection methods simultaneously",
        "normal":    "Flagged by 0 methods",
        "uncertain": "Flagged by 1-2 methods (excluded from precision/recall computation)"
    },
    "justification": (
        "Multi-method agreement provides higher confidence than any single detector. "
        "This approach is conservative: it underestimates recall (misses weak anomalies "
        "that only 1-2 detectors catch) but provides reliable precision estimates. "
        "This is a published technique used in anomaly detection literature when "
        "manual labels are unavailable (Aggarwal, 2017; Goldstein & Uchida, 2016)."
    ),
    "limitations": [
        "Underestimates recall for subtle anomalies detected by only specialized methods",
        "Biased toward anomalies that statistical methods can detect (may miss "
        "purely contextual anomalies caught only by TSA)",
        "Should be replaced with operator-verified labels when >=200 labeled events "
        "are available (via the /feedback API endpoint)"
    ],
    "improvement_path": (
        "The Flask dashboard includes an operator feedback endpoint (/feedback) "
        "that collects human-verified labels. Once 200+ labeled events accumulate, "
        "the consensus-based pseudo-labels can be replaced with supervised ground truth "
        "and detector weights can be optimized via cross-validation."
    )
}

# -------------------------------------------------------------
# EXPORT FOR OTHER SCRIPTS
# -------------------------------------------------------------
ANOMALY_TYPES = [
    'sensor_spike', 'sudden_drain_theft', 'overflow',
    'refill_event', 'slow_leak', 'sensor_freeze'
]

DETECTOR_NAMES = ['MA+SD', 'RoC', 'Adaptive CUSUM', 'STL Residual', 'Prophet']

THRESHOLDS = {
    "masd":  {"window": 50, "n_std": 2.5},
    "roc":   {"drop_threshold": -1.0, "rise_threshold": 1.0},
    "cusum": {"alpha": 0.15, "k": 0.5, "h": 10.0},
    "stl":   {"period": 60, "residual_std_multiplier": 3.0},
    "prophet": {"interval_width": 0.95, "changepoint_prior_scale": 0.05},
    "confidence_tiers": {
        "log":       [0.00, 0.19],
        "dashboard": [0.20, 0.39],
        "alert":     [0.40, 0.69],
        "critical":  [0.70, 1.00]
    }
}


# -------------------------------------------------------------
# PRINT & SAVE REPORTS
# -------------------------------------------------------------
def print_taxonomy():
    """Print the formal anomaly taxonomy."""
    print("=" * 70)
    print("ANOMALY TAXONOMY -- IoT Liquid Tank Monitoring")
    print("=" * 70)
    print("Reference: Chandola et al. (2009) classification framework\n")

    for category, info in ANOMALY_TAXONOMY.items():
        print(f"\n{'-' * 70}")
        print(f"  CATEGORY: {category.upper()} ANOMALY")
        print(f"{'-' * 70}")
        print(f"  Definition  : {info['description']}")
        print(f"  Mathematical: {info['mathematical_definition']}")
        for stype, sinfo in info['subtypes'].items():
            print(f"\n    > {stype}")
            print(f"      Description : {sinfo['description']}")
            print(f"      Detection   : {sinfo['detection_method']}")
            print(f"      Criteria    : {sinfo['criteria']}")
            print(f"      Example     : {sinfo['example']}")

    print(f"\n{'=' * 70}")


def print_parameter_rationale():
    """Print the threshold justification table."""
    print("\n" + "=" * 70)
    print("PARAMETER RATIONALE -- Threshold Justification")
    print("=" * 70)

    for method, params in PARAMETER_RATIONALE.items():
        print(f"\n{'-' * 70}")
        print(f"  {method}")
        print(f"{'-' * 70}")
        for pname, pinfo in params.items():
            if not isinstance(pinfo, dict) or 'value' not in pinfo:
                # Skip nested dicts like 'tiers' that don't have value/rationale
                continue
            val = pinfo['value']
            rationale = pinfo.get('rationale', '')
            unit = pinfo.get('unit', '')
            print(f"\n    {pname} = {val}" + (f"  ({unit})" if unit else ""))
            # Wrap rationale text
            if rationale:
                words = rationale.split()
                line = "      Rationale: "
                for word in words:
                    if len(line) + len(word) > 75:
                        print(line)
                        line = "        "
                    line += word + " "
                print(line)
            if 'note' in pinfo:
                print(f"      Note: {pinfo['note']}")

    print(f"\n{'=' * 70}")


def print_labeling_strategy():
    """Print the labeling strategy documentation."""
    print("\n" + "=" * 70)
    print("LABELING STRATEGY -- Ensemble Consensus Pseudo-Labeling")
    print("=" * 70)
    print(f"\n  Method: {LABELING_STRATEGY['method']}")
    print(f"  {LABELING_STRATEGY['description']}\n")
    print("  Rules:")
    for label, rule in LABELING_STRATEGY['rules'].items():
        print(f"    {label:12s} -> {rule}")
    print(f"\n  Justification: {LABELING_STRATEGY['justification']}")
    print(f"\n  Limitations:")
    for lim in LABELING_STRATEGY['limitations']:
        print(f"    * {lim}")
    print(f"\n  Improvement path: {LABELING_STRATEGY['improvement_path']}")
    print(f"\n{'=' * 70}")


def save_reports():
    """Save all methodology reports to output/."""

    # Taxonomy report
    with open("output/anomaly_taxonomy.txt", "w", encoding="utf-8") as f:
        f.write("ANOMALY TAXONOMY -- IoT Liquid Tank Monitoring\n")
        f.write("=" * 70 + "\n")
        f.write("Reference: Chandola et al. (2009) classification\n\n")
        for cat, info in ANOMALY_TAXONOMY.items():
            f.write(f"\n{cat.upper()} ANOMALY\n")
            f.write(f"  Definition: {info['description']}\n")
            f.write(f"  Math: {info['mathematical_definition']}\n")
            for st, si in info['subtypes'].items():
                f.write(f"\n  > {st}\n")
                f.write(f"    Description: {si['description']}\n")
                f.write(f"    Detection:   {si['detection_method']}\n")
                f.write(f"    Criteria:    {si['criteria']}\n")
                f.write(f"    Example:     {si['example']}\n")

    # Parameter rationale
    with open("output/parameter_rationale.txt", "w", encoding="utf-8") as f:
        f.write("PARAMETER RATIONALE -- Threshold Justification\n")
        f.write("=" * 70 + "\n")
        for method, params in PARAMETER_RATIONALE.items():
            f.write(f"\n{method}\n" + "-" * 40 + "\n")
            for pn, pi in params.items():
                if not isinstance(pi, dict) or 'value' not in pi:
                    continue
                f.write(f"  {pn} = {pi['value']}")
                if 'unit' in pi:
                    f.write(f"  ({pi['unit']})")
                f.write(f"\n    Rationale: {pi.get('rationale', '')}\n")
                if 'note' in pi:
                    f.write(f"    Note: {pi['note']}\n")

    # Thresholds JSON (for Flask backend and TinyML)
    with open("models/stat_thresholds.json", "w") as f:
        json.dump(THRESHOLDS, f, indent=2)

    print("[OK] Saved: output/anomaly_taxonomy.txt")
    print("[OK] Saved: output/parameter_rationale.txt")
    print("[OK] Saved: models/stat_thresholds.json")


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    print_taxonomy()
    print_parameter_rationale()
    print_labeling_strategy()
    save_reports()
    print("\n[OK] Anomaly definitions complete -- ready for Step 1 (preprocessing)")
