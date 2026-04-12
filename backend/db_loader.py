"""Load pipeline TSA results into the SQLite database for dashboard viewing."""
import pandas as pd
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "sensor.db")
CSV_PATH = os.path.join(BASE_DIR, "data", "tsa_results.csv")

if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, "data", "statistical_results.csv")
if not os.path.exists(CSV_PATH):
    print("No results CSV found!")
    exit(1)

print(f"Loading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, index_col='timestamp', parse_dates=True)
print(f"  Rows: {len(df)}")

# Connect to DB
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Recreate readings table
c.execute("DROP TABLE IF EXISTS readings")
c.execute("""
    CREATE TABLE readings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT NOT NULL,
        distance_mm REAL,
        level_pct   REAL,
        masd_flag   INTEGER DEFAULT 0,
        roc_flag    INTEGER DEFAULT 0,
        cusum_flag  INTEGER DEFAULT 0,
        if_flag     INTEGER DEFAULT 0,
        stl_flag    INTEGER DEFAULT 0,
        prophet_flag INTEGER DEFAULT 0,
        n_methods   INTEGER DEFAULT 0,
        conf_score  REAL DEFAULT 0,
        conf_tier   TEXT DEFAULT 'log',
        anomaly_class TEXT DEFAULT 'normal',
        operator_label TEXT DEFAULT 'unreviewed',
        flow_rate   REAL DEFAULT 0,
        flow_direction TEXT DEFAULT 'stable'
    )
""")

# Insert data
inserted = 0
for ts, row in df.iterrows():
    dist_mm = row.get('distance_mm', 0)
    level   = row.get('level_pct', 0)
    masd    = int(row.get('masd_flag', False))
    roc     = int(row.get('roc_flag', False))
    cusum   = int(row.get('cusum_flag', False))
    stl     = int(row.get('stl_flag', False))
    prophet = int(row.get('prophet_flag', False))
    n_meth  = int(row.get('n_methods_total', masd + roc + cusum + stl + prophet))
    
    score_col = 'conf_score_final' if 'conf_score_final' in row.index else 'conf_score'
    tier_col  = 'conf_tier_final' if 'conf_tier_final' in row.index else 'conf_tier'
    cls_col   = 'anomaly_class' if 'anomaly_class' in row.index else None
    
    score = float(row.get(score_col, 0))
    tier  = str(row.get(tier_col, 'log'))
    cls   = str(row.get(cls_col, 'normal')) if cls_col else 'normal'
    
    # Flow rate from RoC
    flow = float(row.get('roc_val', 0)) if 'roc_val' in row.index else 0
    direction = 'drain' if flow > 0.3 else 'fill' if flow < -0.3 else 'stable'
    
    c.execute("""
        INSERT INTO readings 
        (timestamp, distance_mm, level_pct, masd_flag, roc_flag, cusum_flag, 
         if_flag, stl_flag, prophet_flag, n_methods, conf_score, conf_tier, 
         anomaly_class, flow_rate, flow_direction)
        VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ts.strftime('%Y-%m-%d %H:%M:%S'), dist_mm, level, masd, roc, cusum,
          stl, prophet, n_meth, score, tier, cls, flow, direction))
    inserted += 1

# Create alerts table
c.execute("DROP TABLE IF EXISTS alerts")
c.execute("""
    CREATE TABLE alerts (
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

# Insert alerts for high-confidence anomalies
alert_df = df[df.get('n_methods_total', df.get('n_methods', pd.Series(dtype=int))) >= 3]
for ts, row in alert_df.iterrows():
    score_col = 'conf_score_final' if 'conf_score_final' in row.index else 'conf_score'
    tier_col  = 'conf_tier_final' if 'conf_tier_final' in row.index else 'conf_tier'
    c.execute("""
        INSERT INTO alerts (timestamp, anomaly_class, conf_score, conf_tier, distance_mm, level_pct)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (ts.strftime('%Y-%m-%d %H:%M:%S'),
          str(row.get('anomaly_class', 'anomaly')),
          float(row.get(score_col, 0.6)),
          str(row.get(tier_col, 'alert')),
          float(row.get('distance_mm', 0)),
          float(row.get('level_pct', 0))))

conn.commit()
conn.close()

print(f"  Inserted {inserted} readings into SQLite")
print(f"  Inserted {len(alert_df)} alerts")
print(f"  Database: {DB_PATH}")
print("Done!")
