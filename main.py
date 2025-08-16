import requests
import pandas as pd
from pathlib import Path

# ---- paths (robust regardless of where you run) ----
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "output"   # <-- add this
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True) 

# EXTRACT → NYC Transportation Counts
URL = "https://data.cityofnewyork.us/resource/7ym2-wayt.json"
params = {"$limit": 50000}
resp = requests.get(URL, params=params, timeout=30)
resp.raise_for_status()
df = pd.DataFrame(resp.json())


# ---- save ----
out_path = DATA_DIR / "nyc_transportation.csv"
df.to_csv(out_path, index=False)
print("Saved to:", out_path)



# ----------------------------
# 2) TRANSFORM

# a) normalize column names
expected = [
    "requestid","boro","yr","m","d","hh","mm","vol",
    "segmentid","wktgeom","street","fromst","tost","direction"
]
missing = [c for c in expected if c not in df.columns]
if missing:
    raise ValueError(f"API missing expected columns: {missing}")

# --- 2) Rename to readable snake_case ---
rename_map = {
    "requestid": "request_id",
    "boro":      "borough",
    "yr":        "year",
    "m":         "month",
    "d":         "day",
    "hh":        "hour",
    "mm":        "minute",
    "vol":       "volume",
    "segmentid": "segment_id",
    "wktgeom":   "wkt_geom",
    "street":    "street",
    "fromst":    "from_street",
    "tost":      "to_street",
    "direction": "direction"
}
df = df.rename(columns=rename_map)

# --- 3) Type casting ---
# integers that may contain nulls -> use pandas nullable Int64
int_cols = ["request_id","year","month","day","hour","minute","segment_id","volume"]
for c in int_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

# strings: strip whitespace
str_cols = ["borough","street","from_street","to_street","direction","wkt_geom"]
for c in str_cols:
    df[c] = df[c].astype(str).str.strip()

# --- 4) Normalize direction codes ---
# NB/SB/EB/WB -> full words (fallback to original if something else appears)
dir_map = {"NB":"northbound","SB":"southbound","EB":"eastbound","WB":"westbound"}
df["direction"] = df["direction"].str.upper().map(dir_map).fillna(df["direction"])

# --- 5) Build datetime (15-min buckets) and year_month ---

dt = pd.to_datetime(
    dict(
        year=df["year"],
        month=df["month"],
        day=df["day"],
        hour=df["hour"],
        minute=df["minute"]
    ),
    errors="coerce"
)
df["count_datetime"] = dt
df["year_month"] = pd.to_datetime(
    df["year"].astype("Int64").astype("string") + "-" +
    df["month"].astype("Int64").astype("string").str.zfill(2) + "-01",
    errors="coerce"
)

# --- 6) Basic data hygiene ---
# Drop impossible volumes (negative) and exact duplicates
before = len(df)
df = df[df["volume"].isna() | (df["volume"] >= 0)]
df = df.drop_duplicates()
after = len(df)

print(f"🧹 Transform complete. Dropped {before - after} rows (dupes/invalid).")
print("Columns now:", df.columns.tolist())

# ----------------------------
# 3) LOAD → SQLite

import sqlite3

DB_PATH = DATA_DIR / "nyc_transportation.db"
TABLE   = "traffic_counts"

# write to SQLite (replace table each run)
with sqlite3.connect(DB_PATH) as conn:
    df.to_sql(TABLE, conn, if_exists="replace", index=False)
    print(f"🗄️  Loaded {len(df):,} rows into {DB_PATH.name}.{TABLE}")

    # create helpful indexes (skip if a column is missing)
    index_ddls = [
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_borough       ON {TABLE}(borough)",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_year_month    ON {TABLE}(year, month)",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_count_dt      ON {TABLE}(count_datetime)",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_segment       ON {TABLE}(segment_id)",
    ]
    for ddl in index_ddls:
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass

    # quick sanity checks - 50,000 rows
    total_rows = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    print("Rows in table:", total_rows)

 # ----------------------------
# 4) ANALYZE → Top 10 busiest segments

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

DB_PATH = DATA_DIR / "nyc_transportation.db"
TABLE   = "traffic_counts"

top_sql = f"""
    SELECT
        segment_id,
        street,
        from_street,
        to_street,
        borough,
        direction,
        SUM(volume)          AS total_volume,
        COUNT(*)             AS n_observations,
        MIN(count_datetime)  AS first_seen,
        MAX(count_datetime)  AS last_seen
    FROM {TABLE}
    WHERE segment_id IS NOT NULL
      AND volume IS NOT NULL
      AND volume >= 0
    GROUP BY
        segment_id, street, from_street, to_street, borough, direction
    ORDER BY total_volume DESC
    LIMIT 10;
"""

with sqlite3.connect(DB_PATH) as conn:
    top10 = pd.read_sql_query(top_sql, conn, parse_dates=["first_seen", "last_seen"])

# Save as CSV
top_csv = DATA_DIR / "top10_busiest_segments.csv"
top10.to_csv(top_csv, index=False)
print("Top 10 busiest segments saved ->", top_csv)

import textwrap

if not top10.empty:
    # build compact, wrapped labels
    def wrap(s, width=32):
        s = "" if pd.isna(s) else str(s)
        return "\n".join(textwrap.wrap(s, width=width, break_long_words=False))

    labels = top10.apply(
        lambda r: wrap(
            f"{r.get('street','')} ({r.get('from_street','')}→{r.get('to_street','')}) – "
            f"{r.get('borough','')} {r.get('direction','')}"
        ),
        axis=1,
    )

    # horizontal bar chart (fits long names)
    fig_h = 0.6 * len(top10) + 2  # dynamic height so labels don't overlap
    fig, ax = plt.subplots(figsize=(12, fig_h), layout="constrained")
    y = range(len(top10))
    ax.barh(y, top10["total_volume"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)  # smaller y-axis text
    ax.invert_yaxis()  # largest at top

    ax.set_xlabel("Total Volume")
    ax.set_title("Top 10 Busiest NYC Road Segments (by total volume)")

    for i, v in enumerate(top10["total_volume"]):
        ax.text(v, i, f" {int(v):,}", va="center", fontsize=8)

    out_png = OUT_DIR / "top10_busiest_segments.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
else:
    print(" No results returned - check data.")
