"""
=============================================================================
STEP 6: PRODUCTION FLASK BACKEND
=============================================================================
Complete REST API for the Raspberry Pi edge server.
Receives readings from ESP32, runs full detection pipeline,
stores to SQLite, returns confidence-scored result, sends alerts.

Run:  python 06_flask_backend.py
      Access dashboard: http://localhost:5000

Routes:
  POST /sensor          -- receive reading from ESP32
  GET  /status          -- latest tank status (for dashboard polling)
  GET  /history?hours=N -- last N hours of readings
  GET  /anomalies       -- recent anomaly log
  POST /feedback        -- operator confirms/rejects alert
  GET  /dashboard       -- serve the HTML dashboard
=============================================================================
"""

from flask import Flask, request, jsonify, render_template_string
import sqlite3, json, os, joblib, numpy as np, pandas as pd
from datetime import datetime, timedelta
import threading, warnings
warnings.filterwarnings('ignore')

# Prophet & STL (optional -- graceful fallback if not installed)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("[!] Prophet not installed -- forecasting disabled (pip install prophet)")

try:
    from statsmodels.tsa.seasonal import STL
    STL_AVAILABLE = True
except ImportError:
    STL_AVAILABLE = False
    print("[!] statsmodels not installed -- decomposition disabled (pip install statsmodels)")

app = Flask(__name__)

# -------------------------------------------------------------
# LOAD MODELS & THRESHOLDS (once at startup)
# -------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_DIR, "data/sensor.db")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
TANK_HEIGHT = 250  # mm -- actual tank height
FLOW_STABLE_THRESHOLD = 0.3   # mm/min -- below this = 'stable'
FLOW_SMOOTH_WINDOW    = 5     # readings to smooth flow rate

# Prophet model cache (trained in background)
prophet_model    = None
prophet_forecast = None
prophet_lock     = threading.Lock()
PROPHET_MIN_READINGS = 200    # min readings before training
PROPHET_RETRAIN_MINS = 360    # retrain every 6 hours
last_prophet_train   = None

# Load thresholds
with open(os.path.join(MODELS_DIR, "stat_thresholds.json")) as f:
    THRESHOLDS = json.load(f)

# Legacy IF/LSTM models removed -- replaced by STL + Prophet in pipeline
IF_BUNDLE = None
LSTM_THRESHOLD = None

# Load Prophet seasonal data (for edge-side seasonal correction)
PROPHET_SEASONAL = None
prophet_seasonal_path = os.path.join(MODELS_DIR, "prophet_seasonal.json")
if os.path.exists(prophet_seasonal_path):
    with open(prophet_seasonal_path) as f:
        PROPHET_SEASONAL = json.load(f)
    print("[OK] Prophet seasonal data loaded")

# In-memory rolling buffer for algorithms that need history
BUFFER_SIZE = 200
reading_buffer = []   # list of distance_mm values (most recent last)
cusum_state    = {"pos": 0.0, "neg": 0.0, "baseline": None}

# -------------------------------------------------------------
# DATABASE SETUP
# -------------------------------------------------------------
def get_reference_time():
    """Return the latest reading timestamp in DB (or now() as fallback).
    This ensures the dashboard works with both live and historical data."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT MAX(timestamp) FROM readings").fetchone()
        conn.close()
        if row and row[0]:
            return datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
    except Exception:
        pass
    return datetime.now()

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            distance_mm REAL,
            level_pct   REAL,
            masd_flag   INTEGER DEFAULT 0,
            roc_flag    INTEGER DEFAULT 0,
            cusum_flag  INTEGER DEFAULT 0,
            if_flag     INTEGER DEFAULT 0,
            n_methods   INTEGER DEFAULT 0,
            conf_score  REAL DEFAULT 0,
            conf_tier   TEXT DEFAULT 'log',
            anomaly_class TEXT DEFAULT 'normal',
            operator_label TEXT DEFAULT 'unreviewed'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT,
            anomaly_class TEXT,
            conf_score   REAL,
            conf_tier    TEXT,
            distance_mm  REAL,
            level_pct    REAL,
            acknowledged INTEGER DEFAULT 0,
            feedback     TEXT DEFAULT NULL
        )
    """)
    # Migration: add flow columns if missing
    for col, typedef in [('flow_rate', 'REAL DEFAULT 0'),
                         ('flow_direction', "TEXT DEFAULT 'stable'")]:
        try:
            c.execute(f"ALTER TABLE readings ADD COLUMN {col} {typedef}")
        except Exception:
            pass  # column already exists
    conn.commit()
    conn.close()

init_db()
print("[OK] Database initialized")

# -------------------------------------------------------------
# DETECTION PIPELINE (runs on every incoming reading)
# -------------------------------------------------------------
def run_pipeline(distance_mm, timestamp):
    """
    Full detection pipeline for a single new reading.
    Uses in-memory buffer for rolling stats.
    Returns dict with all flags and final score.
    """
    global cusum_state, reading_buffer

    # 1. Update rolling buffer
    reading_buffer.append(distance_mm)
    if len(reading_buffer) > BUFFER_SIZE:
        reading_buffer.pop(0)
    series = pd.Series(reading_buffer)

    # -- MA+SD ---------------------------------------------
    masd_flag = False
    w   = THRESHOLDS['masd']['window']
    nst = THRESHOLDS['masd']['n_std']
    if len(series) >= w // 2:
        roll_mean = series.rolling(w, min_periods=w//2).mean().iloc[-1]
        roll_std  = series.rolling(w, min_periods=w//2).std().iloc[-1]
        if roll_std and roll_std > 0:
            masd_flag = abs(distance_mm - roll_mean) > nst * roll_std

    # -- RoC -----------------------------------------------
    roc_flag = False
    roc_val  = 0.0
    if len(series) >= 2:
        roc_val  = float(series.iloc[-1] - series.iloc[-2])
        drop_thr = THRESHOLDS['roc']['drop_threshold']
        rise_thr = THRESHOLDS['roc']['rise_threshold']
        roc_flag = (roc_val < drop_thr) or (roc_val > rise_thr)

    # -- Adaptive CUSUM ------------------------------------
    if cusum_state["baseline"] is None:
        cusum_state["baseline"] = distance_mm

    alpha = THRESHOLDS['cusum']['alpha']
    k     = THRESHOLDS['cusum']['k']
    h     = THRESHOLDS['cusum']['h']

    mu = alpha * distance_mm + (1 - alpha) * cusum_state["baseline"]
    cusum_state["baseline"] = mu

    cusum_state["pos"] = max(0, cusum_state["pos"] + (distance_mm - mu - k))
    cusum_state["neg"] = max(0, cusum_state["neg"] + (mu - distance_mm - k))
    cusum_flag = (cusum_state["pos"] > h) or (cusum_state["neg"] > h)
    if cusum_flag:
        cusum_state["pos"] = 0.0
        cusum_state["neg"] = 0.0

    # -- Isolation Forest ----------------------------------
    if_flag = False
    if IF_BUNDLE and len(series) >= 10:
        std10 = float(series.tail(10).std())
        dev10 = distance_mm - float(series.tail(10).mean())
        feat  = np.array([[distance_mm, roc_val, std10, dev10, 0]])
        try:
            feat_scaled = IF_BUNDLE['scaler'].transform(feat)
            if_flag = bool(IF_BUNDLE['model'].predict(feat_scaled)[0] == -1)
        except Exception:
            pass

    # -- Flow Rate Calculation -----------------------------
    flow_rate_raw = roc_val  # mm/min (positive = draining, negative = filling)
    if len(series) >= FLOW_SMOOTH_WINDOW:
        diffs = series.diff().dropna().tail(FLOW_SMOOTH_WINDOW)
        flow_rate_smooth = float(diffs.mean())
    else:
        flow_rate_smooth = flow_rate_raw

    if flow_rate_smooth > FLOW_STABLE_THRESHOLD:
        flow_direction = 'draining'
    elif flow_rate_smooth < -FLOW_STABLE_THRESHOLD:
        flow_direction = 'filling'
    else:
        flow_direction = 'stable'

    # -- Confidence Score ----------------------------------
    n_agree = int(masd_flag) + int(roc_flag) + int(cusum_flag) + int(if_flag)
    conf_score = n_agree / 4.0

    if   conf_score < 0.20: tier = 'log'
    elif conf_score < 0.40: tier = 'dashboard'
    elif conf_score < 0.70: tier = 'alert'
    else:                   tier = 'critical'

    # -- Anomaly Classification ----------------------------
    anomaly_class = 'normal'
    if conf_score > 0:
        std_recent = float(series.tail(20).std()) if len(series) >= 20 else 0
        if std_recent < 0.2:
            anomaly_class = 'sensor_freeze'
        elif masd_flag and not roc_flag and not cusum_flag:
            anomaly_class = 'sensor_spike'
        elif distance_mm < 30:
            anomaly_class = 'overflow'
        elif roc_flag and roc_val < -2.0:
            anomaly_class = 'sudden_drain_theft'
        elif roc_flag and roc_val > 2.0:
            anomaly_class = 'refill_event'
        elif cusum_flag and abs(roc_val) < 1.5:
            anomaly_class = 'slow_leak'
        else:
            anomaly_class = 'unclassified_anomaly'

    level_pct = max(0, min(100, (TANK_HEIGHT - distance_mm) / TANK_HEIGHT * 100))

    return {
        "timestamp"      : timestamp,
        "distance_mm"    : round(distance_mm, 2),
        "level_pct"      : round(level_pct, 1),
        "masd_flag"      : bool(masd_flag),
        "roc_flag"       : bool(roc_flag),
        "cusum_flag"     : bool(cusum_flag),
        "if_flag"        : bool(if_flag),
        "roc_val"        : round(roc_val, 3),
        "n_methods"      : n_agree,
        "conf_score"     : round(conf_score, 3),
        "conf_tier"      : tier,
        "anomaly_class"  : anomaly_class,
        "flow_rate"      : round(flow_rate_smooth, 3),
        "flow_direction" : flow_direction,
    }

# -------------------------------------------------------------
# ALERT SENDER
# -------------------------------------------------------------
def send_alert(result):
    """
    Sends alert based on tier. Extend with:
      - Twilio SMS: from twilio.rest import Client
      - Email: smtplib
      - WhatsApp: twilio.rest with WhatsApp endpoint
    """
    tier = result['conf_tier']
    cls  = result['anomaly_class']
    lvl  = result['level_pct']
    ts   = result['timestamp']

    msg = (f"[{tier.upper()}] Tank Alert @ {ts}\n"
           f"Type: {cls} | Level: {lvl:.1f}% | "
           f"Score: {result['conf_score']:.2f} | "
           f"Methods: {result['n_methods']}")

    print(f"  ALERT -> {msg}")

    # Store alert in DB
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO alerts
        (timestamp, anomaly_class, conf_score, conf_tier, distance_mm, level_pct)
        VALUES (?,?,?,?,?,?)""",
        (ts, cls, result['conf_score'], tier,
         result['distance_mm'], result['level_pct']))
    conn.commit()
    conn.close()

    # -- Uncomment to enable SMS via Twilio ----------------
    # from twilio.rest import Client
    # client = Client(os.environ['TWILIO_SID'], os.environ['TWILIO_TOKEN'])
    # client.messages.create(
    #     body=msg,
    #     from_='+1XXXXXXXXXX',
    #     to='+91XXXXXXXXXX'
    # )

# -------------------------------------------------------------
# API ROUTES
# -------------------------------------------------------------
@app.route('/sensor', methods=['POST'])
def receive_sensor_reading():
    """
    Called by ESP32 every reading.
    Expected JSON: {"distance_mm": 142.5, "device_id": "ESP32-Sensor-01"}
    """
    data = request.json
    if not data or 'distance_mm' not in data:
        return jsonify({"error": "distance_mm required"}), 400

    distance_mm = float(data['distance_mm'])
    timestamp   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Skip init readings
    if distance_mm > 290 or distance_mm < 5:
        return jsonify({"status": "skipped", "reason": "out of range"}), 200

    result = run_pipeline(distance_mm, timestamp)

    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO readings
        (timestamp, distance_mm, level_pct, masd_flag, roc_flag,
         cusum_flag, if_flag, n_methods, conf_score, conf_tier,
         anomaly_class, flow_rate, flow_direction)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (timestamp, result['distance_mm'], result['level_pct'],
         int(result['masd_flag']), int(result['roc_flag']),
         int(result['cusum_flag']), int(result['if_flag']),
         result['n_methods'], result['conf_score'],
         result['conf_tier'], result['anomaly_class'],
         result['flow_rate'], result['flow_direction']))
    conn.commit()
    conn.close()

    # Send alert if above dashboard tier
    if result['conf_tier'] in ('alert', 'critical'):
        send_alert(result)

    return jsonify(result), 200


@app.route('/status', methods=['GET'])
def get_status():
    """Latest tank status -- polled by dashboard every 5s."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT * FROM readings ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "no data"}), 404

    cols = ['id','timestamp','distance_mm','level_pct','masd_flag',
            'roc_flag','cusum_flag','if_flag','n_methods','conf_score',
            'conf_tier','anomaly_class','operator_label']
    return jsonify(dict(zip(cols, row)))


@app.route('/history', methods=['GET'])
def get_history():
    """Last N hours of readings for chart."""
    hours = int(request.args.get('hours', 6))
    since = (get_reference_time() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
    conn  = sqlite3.connect(DB_PATH)
    rows  = conn.execute(
        "SELECT timestamp, distance_mm, level_pct, conf_score, anomaly_class "
        "FROM readings WHERE timestamp > ? ORDER BY id",
        (since,)
    ).fetchall()
    conn.close()
    return jsonify([{"ts": r[0], "dist": r[1], "level": r[2],
                     "score": r[3], "class": r[4]} for r in rows])


@app.route('/anomalies', methods=['GET'])
def get_anomalies():
    """Recent anomaly log."""
    limit = int(request.args.get('limit', 50))
    conn  = sqlite3.connect(DB_PATH)
    rows  = conn.execute("""
        SELECT timestamp, distance_mm, level_pct, conf_score,
               conf_tier, anomaly_class, operator_label
        FROM readings WHERE conf_tier != 'log'
        ORDER BY id DESC LIMIT ?""", (limit,)
    ).fetchall()
    conn.close()
    cols = ['timestamp','distance_mm','level_pct','conf_score',
            'conf_tier','anomaly_class','operator_label']
    return jsonify([dict(zip(cols, r)) for r in rows])


@app.route('/feedback', methods=['POST'])
def operator_feedback():
    """
    Operator marks alert as confirmed anomaly or false alarm.
    This builds the labeled training set for future model improvement.
    POST: {"reading_id": 42, "label": "confirmed" | "false_alarm" | "refill"}
    """
    data  = request.json
    rid   = data.get('reading_id')
    label = data.get('label')
    if not rid or not label:
        return jsonify({"error": "reading_id and label required"}), 400

    valid_labels = ('confirmed', 'false_alarm', 'refill', 'normal')
    if label not in valid_labels:
        return jsonify({"error": f"label must be one of {valid_labels}"}), 400

    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE readings SET operator_label=? WHERE id=?", (label, rid))
    conn.commit()
    labeled_count = conn.execute(
        "SELECT COUNT(*) FROM readings WHERE operator_label != 'unreviewed'"
    ).fetchone()[0]
    conn.close()

    response = {"status": "ok", "labeled_total": labeled_count}
    if labeled_count >= 200:
        response["note"] = "200+ labels available -- consider retraining Random Forest classifier"

    return jsonify(response)


# -------------------------------------------------------------
# PROPHET & STL HELPERS
# -------------------------------------------------------------
def maybe_train_prophet():
    """Train/retrain Prophet model in background if enough data exists."""
    global prophet_model, prophet_forecast, last_prophet_train
    if not PROPHET_AVAILABLE:
        return
    now = datetime.now()
    if last_prophet_train and (now - last_prophet_train).total_seconds() < PROPHET_RETRAIN_MINS * 60:
        return  # too soon to retrain

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, level_pct FROM readings ORDER BY id"
    ).fetchall()
    conn.close()

    if len(rows) < PROPHET_MIN_READINGS:
        return

    df = pd.DataFrame(rows, columns=['ds', 'y'])
    df['ds'] = pd.to_datetime(df['ds'])

    try:
        m = Prophet(daily_seasonality=True, weekly_seasonality=False,
                    changepoint_prior_scale=0.1, seasonality_mode='additive')
        m.fit(df)
        future = m.make_future_dataframe(periods=1440, freq='min')
        fc = m.predict(future)
        with prophet_lock:
            prophet_model = m
            prophet_forecast = fc
            last_prophet_train = now
        print(f"[OK] Prophet retrained on {len(rows)} readings")
    except Exception as e:
        print(f"[X] Prophet training failed: {e}")


def run_stl(hours=6, period=60):
    """Run STL decomposition on recent data. Returns dict or None."""
    if not STL_AVAILABLE:
        return None
    conn = sqlite3.connect(DB_PATH)
    since = (get_reference_time() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
    rows = conn.execute(
        "SELECT timestamp, level_pct FROM readings WHERE timestamp > ? ORDER BY id",
        (since,)
    ).fetchall()
    conn.close()

    if len(rows) < period * 2:
        return None

    df = pd.DataFrame(rows, columns=['ts', 'level'])
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts')
    df = df['level'].resample('1min').mean().interpolate()

    if len(df) < period * 2:
        return None

    try:
        stl = STL(df, period=period, robust=True)
        res = stl.fit()
        return {
            "timestamps": [t.strftime('%Y-%m-%d %H:%M') for t in df.index],
            "observed":   [round(v, 2) for v in df.values],
            "trend":      [round(v, 2) for v in res.trend.values],
            "seasonal":   [round(v, 2) for v in res.seasonal.values],
            "residual":   [round(v, 2) for v in res.resid.values],
        }
    except Exception:
        return None


# -------------------------------------------------------------
# NEW API ROUTES -- FLOW & TRENDS
# -------------------------------------------------------------
@app.route('/flow', methods=['GET'])
def get_flow():
    """Flow rate history for the last N hours."""
    hours = int(request.args.get('hours', 6))
    since = (get_reference_time() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, flow_rate, flow_direction, level_pct "
        "FROM readings WHERE timestamp > ? ORDER BY id", (since,)
    ).fetchall()
    conn.close()
    return jsonify([{"ts": r[0], "flow_rate": r[1], "direction": r[2],
                     "level": r[3]} for r in rows])


@app.route('/trends', methods=['GET'])
def get_trends():
    """Hourly usage trends -- avg flow per hour of day."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, flow_rate, level_pct FROM readings ORDER BY id"
    ).fetchall()
    conn.close()

    if not rows:
        return jsonify({"error": "no data"}), 404

    df = pd.DataFrame(rows, columns=['ts', 'flow_rate', 'level'])
    df['ts'] = pd.to_datetime(df['ts'])
    df['hour'] = df['ts'].dt.hour

    hourly = df.groupby('hour').agg(
        avg_flow=('flow_rate', 'mean'),
        avg_level=('level', 'mean'),
        readings=('flow_rate', 'count')
    ).reset_index()

    hourly_list = []
    for _, row in hourly.iterrows():
        af = round(row['avg_flow'], 3)
        pattern = 'high_drain' if af > 0.5 else 'drain' if af > 0.1 else \
                  'high_fill' if af < -0.5 else 'fill' if af < -0.1 else 'stable'
        hourly_list.append({
            "hour": int(row['hour']), "avg_flow": af,
            "avg_level": round(row['avg_level'], 1), "pattern": pattern,
            "readings": int(row['readings'])
        })

    peak_hour = max(hourly_list, key=lambda x: x['avg_flow'])
    low_hour  = min(hourly_list, key=lambda x: x['avg_flow'])

    # Prophet daily component (if available)
    prophet_daily = None
    with prophet_lock:
        if prophet_forecast is not None:
            fc = prophet_forecast.copy()
            fc['hour'] = fc['ds'].dt.hour
            prophet_daily = fc.groupby('hour')['daily'].mean().round(2).to_dict()

    return jsonify({
        "hourly_avg": hourly_list,
        "peak_usage_hour": peak_hour['hour'],
        "lowest_usage_hour": low_hour['hour'],
        "prophet_daily": prophet_daily,
    })


@app.route('/flow_summary', methods=['GET'])
def get_flow_summary():
    """Current flow summary for dashboard cards."""
    conn = sqlite3.connect(DB_PATH)
    latest = conn.execute(
        "SELECT flow_rate, flow_direction, level_pct, distance_mm "
        "FROM readings ORDER BY id DESC LIMIT 1"
    ).fetchone()

    since_1h = (get_reference_time() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    avg_row = conn.execute(
        "SELECT AVG(flow_rate) FROM readings WHERE timestamp > ?", (since_1h,)
    ).fetchone()
    conn.close()

    if not latest:
        return jsonify({"error": "no data"}), 404

    flow_rate, direction, level_pct, dist_mm = latest
    avg_1h = round(avg_row[0], 3) if avg_row and avg_row[0] else 0

    # Estimate time to empty/full
    est_empty = est_full = None
    if flow_rate and flow_rate > 0.01:
        remaining_mm = TANK_HEIGHT - dist_mm  # mm of water left
        mins = remaining_mm / flow_rate
        if mins > 0:
            est_empty = f"{int(mins // 60)}h {int(mins % 60)}m"
    elif flow_rate and flow_rate < -0.01:
        mm_to_full = dist_mm
        mins = abs(mm_to_full / flow_rate)
        if mins > 0:
            est_full = f"{int(mins // 60)}h {int(mins % 60)}m"

    # Trend: accelerating or decelerating?
    conn2 = sqlite3.connect(DB_PATH)
    recent = conn2.execute(
        "SELECT flow_rate FROM readings ORDER BY id DESC LIMIT 10"
    ).fetchall()
    conn2.close()
    trend = 'stable'
    if len(recent) >= 5:
        first_half = np.mean([r[0] for r in recent[5:]])
        second_half = np.mean([r[0] for r in recent[:5]])
        if second_half > first_half + 0.1:
            trend = 'accelerating'
        elif second_half < first_half - 0.1:
            trend = 'decelerating'

    return jsonify({
        "current_flow_rate": flow_rate,
        "flow_direction": direction,
        "est_time_to_empty": est_empty,
        "est_time_to_full": est_full,
        "last_1h_avg_flow": avg_1h,
        "trend": trend,
        "level_pct": level_pct,
    })


@app.route('/forecast', methods=['GET'])
def get_forecast():
    """Prophet forecast for the next N hours."""
    hours = int(request.args.get('hours', 24))
    # Trigger training if needed
    maybe_train_prophet()

    with prophet_lock:
        if prophet_forecast is None:
            return jsonify({"error": "Not enough data for forecast",
                            "min_readings": PROPHET_MIN_READINGS}), 200

        now = datetime.now()
        future_mask = prophet_forecast['ds'] > now
        fc = prophet_forecast[future_mask].head(hours * 60)
        return jsonify({
            "forecast": [
                {"ts": row['ds'].strftime('%Y-%m-%d %H:%M'),
                 "predicted_level": round(row['yhat'], 1),
                 "upper": round(row['yhat_upper'], 1),
                 "lower": round(row['yhat_lower'], 1)}
                for _, row in fc.iterrows()
            ],
            "model_trained_on": last_prophet_train.strftime('%Y-%m-%d %H:%M') if last_prophet_train else None,
        })


@app.route('/decomposition', methods=['GET'])
def get_decomposition():
    """STL decomposition of recent data."""
    hours = int(request.args.get('hours', 6))
    result = run_stl(hours=hours)
    if result is None:
        return jsonify({"error": "Not enough data for decomposition"}), 200
    return jsonify(result)


@app.route('/report')
def report():
    report_path = os.path.join(BASE_DIR, 'output', 'project_report.html')
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            return f.read()
    return 'Report not found', 404


@app.route('/dashboard')
def dashboard():
    """Enhanced dashboard with flow analysis, trends & forecasting."""
    return render_template_string(DASHBOARD_HTML)


# -------------------------------------------------------------
# DASHBOARD HTML (served by Flask, no separate frontend needed)
# -------------------------------------------------------------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tank Monitor Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f1117; color: #e0e0e0; padding: 20px; }
  h1   { color: #4A9EFF; margin-bottom: 8px; }
  h3   { color: #4A9EFF; margin-bottom: 14px; }
  .subtitle { color: #888; font-size: 0.85em; margin-bottom: 20px; }
  .section-title { color: #00E676; margin: 24px 0 12px; font-size: 1.1em; }
  .cards { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
  .card  { background: #1a1d27; border-radius: 10px; padding: 16px;
           flex: 1; min-width: 140px; border-left: 4px solid #4A9EFF; }
  .card.warn  { border-color: #FFA500; }
  .card.alert { border-color: #FF4444; }
  .card.flow-drain  { border-color: #FF6B6B; }
  .card.flow-fill   { border-color: #00E676; }
  .card.flow-stable { border-color: #888; }
  .card-val  { font-size: 1.8em; font-weight: bold; color: #4A9EFF; }
  .card.warn .card-val  { color: #FFA500; }
  .card.alert .card-val { color: #FF4444; }
  .card.flow-drain .card-val  { color: #FF6B6B; }
  .card.flow-fill .card-val   { color: #00E676; }
  .card.flow-stable .card-val { color: #888; }
  .card-lbl  { font-size: 0.75em; color: #888; margin-top: 4px; }
  .chart-box { background: #1a1d27; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media(max-width:900px) { .grid-2 { grid-template-columns: 1fr; } }
  table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  th    { background: #252836; padding: 8px 12px; text-align: left; color: #888; }
  td    { padding: 8px 12px; border-bottom: 1px solid #252836; }
  .tier-log       { color: #888; }
  .tier-dashboard { color: #FFA500; font-weight: bold; }
  .tier-alert     { color: #FF8C00; font-weight: bold; }
  .tier-critical  { color: #FF4444; font-weight: bold; }
  .btn { padding: 4px 10px; border: none; border-radius: 4px; cursor: pointer;
         font-size: 0.8em; margin: 0 2px; }
  .btn-ok  { background: #2d6a4f; color: white; }
  .btn-no  { background: #6b2737; color: white; }
  .arrow { font-size: 1.4em; }
  /* Navigation bar */
  .navbar { background: #1a1d27; border-radius: 10px; padding: 10px 16px;
            margin-bottom: 20px; display: flex; align-items: center;
            gap: 8px; flex-wrap: wrap; border-bottom: 2px solid #252836;
            position: sticky; top: 0; z-index: 100; }
  .navbar .nav-label { color: #888; font-size: 0.8em; margin-right: 4px; font-weight: bold; }
  .nav-btn { padding: 6px 14px; border: 1px solid #333; border-radius: 6px;
             cursor: pointer; font-size: 0.8em; background: #252836; color: #ccc;
             text-decoration: none; transition: all 0.2s; display: inline-block; }
  .nav-btn:hover { background: #4A9EFF; color: white; border-color: #4A9EFF; }
  .nav-btn.active { background: #4A9EFF; color: white; border-color: #4A9EFF; }
  .nav-btn.api { background: #1e2233; border-color: #444; color: #aaa; font-family: monospace; }
  .nav-btn.api:hover { background: #FF8C00; color: white; border-color: #FF8C00; }
  .nav-sep { width: 1px; height: 24px; background: #333; margin: 0 4px; }
</style>
</head>
<body>

<!-- Navigation Bar -->
<nav class="navbar">
  <span class="nav-label">IoT Tank Monitor</span>
  <a class="nav-btn active" href="/dashboard">Dashboard</a>
</nav>

<h1>🛢 Liquid Tank Monitor</h1>
<p class="subtitle">Real-time anomaly detection + flow trend analysis</p>

<!-- Status Cards -->
<div class="cards" id="cards">
  <div class="card"><div class="card-val" id="level">--</div><div class="card-lbl">Tank Level %</div></div>
  <div class="card"><div class="card-val" id="dist">--</div><div class="card-lbl">Distance (mm)</div></div>
  <div class="card" id="card-score"><div class="card-val" id="score">--</div><div class="card-lbl">Confidence Score</div></div>
  <div class="card" id="card-tier"><div class="card-val" id="tier">--</div><div class="card-lbl">Alert Tier</div></div>
  <div class="card"><div class="card-val" id="cls">--</div><div class="card-lbl">Anomaly Class</div></div>
  <div class="card"><div class="card-val" id="ts" style="font-size:1em">--</div><div class="card-lbl">Last Reading</div></div>
</div>

<!-- Flow Cards -->
<h3 class="section-title">📊 Flow Analysis</h3>
<div class="cards" id="flow-cards">
  <div class="card flow-stable" id="fc-rate"><div class="card-val" id="flow-rate">--</div><div class="card-lbl">Flow Rate (mm/min)</div></div>
  <div class="card flow-stable" id="fc-dir"><div class="card-val" id="flow-dir"><span class="arrow">-></span> --</div><div class="card-lbl">Flow Direction</div></div>
  <div class="card"><div class="card-val" id="flow-tte">--</div><div class="card-lbl">Est. Time to Empty</div></div>
  <div class="card"><div class="card-val" id="flow-ttf">--</div><div class="card-lbl">Est. Time to Full</div></div>
  <div class="card"><div class="card-val" id="flow-avg1h">--</div><div class="card-lbl">Avg Flow (1h)</div></div>
  <div class="card"><div class="card-val" id="flow-trend">--</div><div class="card-lbl">Trend</div></div>
</div>

<!-- Charts -->
<div class="chart-box">
  <h3>Tank Level -- Last 6 Hours</h3>
  <canvas id="chart" height="80"></canvas>
</div>

<div class="grid-2">
  <div class="chart-box">
    <h3>Flow Rate -- Last 6 Hours</h3>
    <canvas id="flowChart" height="120"></canvas>
  </div>
  <div class="chart-box">
    <h3>Hourly Usage Pattern</h3>
    <canvas id="trendChart" height="120"></canvas>
  </div>
</div>

<!-- Anomaly Log -->
<div class="chart-box">
  <h3>Recent Anomaly Log</h3>
  <table><thead><tr>
    <th>Time</th><th>Level%</th><th>Distance</th>
    <th>Score</th><th>Tier</th><th>Class</th><th>Feedback</th>
  </tr></thead><tbody id="anom-body"></tbody></table>
</div>

<script>
let chart, flowChartObj, trendChartObj;

function buildChart(labels, levels, scores) {
  const ctx = document.getElementById('chart').getContext('2d');
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Level %', data: levels, borderColor: '#4A9EFF',
          backgroundColor: 'rgba(74,158,255,0.1)', tension: 0.2,
          yAxisID: 'y', pointRadius: 0, fill: true },
        { label: 'Conf. Score', data: scores, borderColor: '#FF8C00',
          borderDash: [4,2], tension: 0.1, yAxisID: 'y2', pointRadius: 0 }
      ]
    },
    options: {
      animation: false,
      scales: {
        x:  { ticks: { color: '#888', maxTicksLimit: 8 }, grid: { color: '#222' } },
        y:  { ticks: { color: '#4A9EFF' }, min: 0, max: 100,
               title: { display: true, text: 'Level %', color: '#4A9EFF' } },
        y2: { position: 'right', ticks: { color: '#FF8C00' }, min: 0, max: 1,
               title: { display: true, text: 'Conf. Score', color: '#FF8C00' },
               grid: { drawOnChartArea: false } }
      },
      plugins: { legend: { labels: { color: '#ccc' } } }
    }
  });
}

function buildFlowChart(labels, rates) {
  const ctx = document.getElementById('flowChart').getContext('2d');
  if (flowChartObj) flowChartObj.destroy();
  const colors = rates.map(v => v > 0.3 ? '#FF6B6B' : v < -0.3 ? '#00E676' : '#888');
  flowChartObj = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{
      label: 'Flow Rate (mm/min)', data: rates, borderColor: '#FF6B6B',
      backgroundColor: 'rgba(255,107,107,0.1)', tension: 0.3, pointRadius: 0, fill: true,
      segment: { borderColor: ctx2 => {
        const v = ctx2.p1.parsed.y;
        return v > 0.3 ? '#FF6B6B' : v < -0.3 ? '#00E676' : '#888';
      }}
    }] },
    options: {
      animation: false,
      scales: {
        x: { ticks: { color: '#888', maxTicksLimit: 8 }, grid: { color: '#222' } },
        y: { ticks: { color: '#FF6B6B' },
             title: { display: true, text: 'mm/min', color: '#FF6B6B' } }
      },
      plugins: {
        legend: { labels: { color: '#ccc' } },
        annotation: { annotations: {
          zero: { type: 'line', yMin: 0, yMax: 0, borderColor: '#555', borderWidth: 1 }
        }}
      }
    }
  });
}

function buildTrendChart(hours, flows) {
  const ctx = document.getElementById('trendChart').getContext('2d');
  if (trendChartObj) trendChartObj.destroy();
  const bgColors = flows.map(v => v > 0.5 ? '#FF4444' : v > 0.1 ? '#FF8C00' :
                                     v < -0.5 ? '#00E676' : v < -0.1 ? '#4A9EFF' : '#555');
  trendChartObj = new Chart(ctx, {
    type: 'bar',
    data: { labels: hours.map(h => h + ':00'), datasets: [{
      label: 'Avg Flow (mm/min)', data: flows, backgroundColor: bgColors,
      borderRadius: 4
    }] },
    options: {
      animation: false,
      scales: {
        x: { ticks: { color: '#888' }, grid: { color: '#222' } },
        y: { ticks: { color: '#FF8C00' },
             title: { display: true, text: 'mm/min', color: '#FF8C00' } }
      },
      plugins: { legend: { labels: { color: '#ccc' } } }
    }
  });
}

async function refreshStatus() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    document.getElementById('level').textContent = d.level_pct?.toFixed(1) + '%';
    document.getElementById('dist').textContent  = d.distance_mm?.toFixed(1) + ' mm';
    document.getElementById('score').textContent = d.conf_score?.toFixed(2);
    document.getElementById('tier').textContent  = (d.conf_tier || 'log').toUpperCase();
    document.getElementById('cls').textContent   = (d.anomaly_class || '--').replace('_',' ');
    document.getElementById('ts').textContent    = d.timestamp?.slice(11,16) || '--';
    const sc = parseFloat(d.conf_score || 0);
    const ct = document.getElementById('card-score');
    ct.className = 'card' + (sc >= 0.7 ? ' alert' : sc >= 0.4 ? ' warn' : '');
  } catch(e) {}
}

async function refreshFlowSummary() {
  try {
    const r = await fetch('/flow_summary');
    const d = await r.json();
    if (d.error) return;
    const fr = d.current_flow_rate || 0;
    document.getElementById('flow-rate').textContent = fr.toFixed(3);
    const arrows = {draining:'v', filling:'^', stable:'->'};
    const dir = d.flow_direction || 'stable';
    document.getElementById('flow-dir').innerHTML =
      '<span class="arrow">' + (arrows[dir]||'->') + '</span> ' + dir;
    document.getElementById('flow-tte').textContent = d.est_time_to_empty || '--';
    document.getElementById('flow-ttf').textContent = d.est_time_to_full || '--';
    document.getElementById('flow-avg1h').textContent = (d.last_1h_avg_flow||0).toFixed(3);
    document.getElementById('flow-trend').textContent = d.trend || 'stable';
    // Color flow cards
    const cls = dir === 'draining' ? 'flow-drain' : dir === 'filling' ? 'flow-fill' : 'flow-stable';
    document.getElementById('fc-rate').className = 'card ' + cls;
    document.getElementById('fc-dir').className = 'card ' + cls;
  } catch(e) {}
}

async function refreshChart() {
  try {
    const r = await fetch('/history?hours=6');
    const d = await r.json();
    buildChart(d.map(x => x.ts.slice(11,16)),
               d.map(x => x.level), d.map(x => x.score));
  } catch(e) {}
}

async function refreshFlowChart() {
  try {
    const r = await fetch('/flow?hours=6');
    const d = await r.json();
    if (d.length) buildFlowChart(d.map(x => x.ts.slice(11,16)), d.map(x => x.flow_rate));
  } catch(e) {}
}

async function refreshTrends() {
  try {
    const r = await fetch('/trends');
    const d = await r.json();
    if (d.hourly_avg) {
      const all24 = Array.from({length:24}, (_,i) => {
        const found = d.hourly_avg.find(h => h.hour === i);
        return found ? found.avg_flow : 0;
      });
      buildTrendChart(Array.from({length:24}, (_,i) => i), all24);
    }
  } catch(e) {}
}

async function refreshAnomalies() {
  try {
    const r = await fetch('/anomalies?limit=20');
    const rows = await r.json();
    const tc = {'log':'tier-log','dashboard':'tier-dashboard',
                'alert':'tier-alert','critical':'tier-critical'};
    document.getElementById('anom-body').innerHTML = rows.map(row => `
      <tr>
        <td>${row.timestamp.slice(11,16)}</td>
        <td>${row.level_pct?.toFixed(1)}%</td>
        <td>${row.distance_mm?.toFixed(1)} mm</td>
        <td>${row.conf_score?.toFixed(2)}</td>
        <td class="${tc[row.conf_tier] || ''}">${row.conf_tier?.toUpperCase()}</td>
        <td>${(row.anomaly_class || '').replace(/_/g,' ')}</td>
        <td>
          <button class="btn btn-ok"  onclick="feedback(${row.id||0},'confirmed')">[OK]</button>
          <button class="btn btn-no"  onclick="feedback(${row.id||0},'false_alarm')">[X]</button>
        </td>
      </tr>`).join('');
  } catch(e) {}
}

async function feedback(id, label) {
  await fetch('/feedback', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({reading_id: id, label})});
  refreshAnomalies();
}

function refresh() {
  refreshStatus(); refreshFlowSummary(); refreshChart();
  refreshFlowChart(); refreshAnomalies();
}
refresh();
refreshTrends();  // trends update less frequently
setInterval(refresh, 5000);
setInterval(refreshTrends, 60000);  // trends every 60s
</script>
</body>
</html>"""


# -------------------------------------------------------------
# RUN
# -------------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "="*65)
    print("Flask Tank Monitor Backend")
    print("="*65)
    print(f"  Dashboard  : http://localhost:5000/dashboard")
    print(f"  API status : http://localhost:5000/status")
    print(f"  Database   : {DB_PATH}")
    print("="*65)
    app.run(host='0.0.0.0', port=5000, debug=False)
