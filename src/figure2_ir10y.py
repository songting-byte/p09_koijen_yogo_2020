"""figure2_ir10y.py
=========================
Pull 10-year government bond yields (IRLT) from the OECD SDMX API.
Used for the Figure 2 y-axis and x-axis price adjustment.

Note on Germany (DEU):
  DEU is not available in DF_FINMARK for IRLT. EA19 is used as a proxy,
  then renamed to DEU. Germany's 10Y yield is the euro-area benchmark
  and is virtually identical to the EA19 IRLT series.

Output (written to DATA_DIR):
  IR10Y_PANEL_FILE    long-format parquet  (TIME_PERIOD, REF_AREA, IR10Y)
  IR10Y_WIDE_FILE     wide-format parquet  (TIME_PERIOD × country)
                      column "DEU" is sourced from EA19

Configuration (via settings.py / .env / CLI):
  DATA_DIR              output directory     (default: _data/)
  OECD_START_PERIOD     start year           (default: "2003")
  OECD_END_PERIOD       end year             (default: "2020")
  IR10Y_REF_AREAS       comma-separated list (default: USA,EA19,JPN,CHE,GBR)
  IR10Y_PANEL_FILE      output filename      (default: oecd_ir10y_panel.parquet)
  IR10Y_WIDE_FILE       output filename      (default: figure2_yaxis.parquet)

Dependencies: pip install requests pandas pyarrow python-decouple
"""

import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from settings import config

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR   = config("DATA_DIR")
START_YEAR = config("OECD_START_PERIOD", default="2003", cast=str)
END_YEAR   = config("OECD_END_PERIOD",   default="2020", cast=str)

START = f"{START_YEAR}-01"
END   = f"{END_YEAR}-12"

_DEFAULT_REF_AREAS = "USA,EA19,JPN,CHE,GBR"
IR10Y_REF_AREAS    = config("IR10Y_REF_AREAS", default=_DEFAULT_REF_AREAS, cast=str)
REF_AREAS          = [c.strip() for c in IR10Y_REF_AREAS.split(",")]

IR10Y_PANEL_FILE = config("IR10Y_PANEL_FILE", default="oecd_ir10y_panel.parquet", cast=str)
IR10Y_WIDE_FILE  = config("IR10Y_WIDE_FILE",  default="figure2_yaxis.parquet",    cast=str)

# EA19 is fetched and then renamed to DEU in the output
RENAME_MAP = {"EA19": "DEU"}

# ── OECD endpoint ─────────────────────────────────────────────────────────────

FLOW_REF = "OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0"
BASE_URL  = f"https://sdmx.oecd.org/public/rest/data/{FLOW_REF}"

# ── Fetch helpers ─────────────────────────────────────────────────────────────

def fetch_one(ref_area: str) -> pd.DataFrame:
    """Fetch IRLT monthly series for a single country/region."""
    key = f"{ref_area}.M.IRLT.PA....."
    url = (
        f"{BASE_URL}/{key}"
        f"?startPeriod={START}&endPeriod={END}"
        f"&dimensionAtObservation=AllDimensions"
        f"&format=csvfilewithlabels"
    )
    headers = {
        "Accept":     "text/csv",
        "User-Agent": "python-requests (OECD SDMX client)",
    }
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code in (403, 429):
        raise RuntimeError(
            f"{ref_area} blocked or rate-limited (HTTP {r.status_code}). "
            "Try reducing request frequency."
        )
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def pull_ir10y(
    ref_areas: list[str] = REF_AREAS,
    max_retries: int = 3,
    sleep_between: float = 1.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull IRLT for all requested areas and return (panel, wide) DataFrames.

    EA19 is automatically renamed to DEU in both outputs.

    Returns
    -------
    panel : TIME_PERIOD, REF_AREA, IR10Y
    wide  : TIME_PERIOD × country  (DEU column sourced from EA19)
    """
    all_df = []
    for area in ref_areas:
        for attempt in range(max_retries):
            try:
                df = fetch_one(area)
                df["REF_AREA_REQ"] = RENAME_MAP.get(area, area)
                all_df.append(df)
                print(f"OK: {area} → {RENAME_MAP.get(area, area)}, rows={len(df)}")
                time.sleep(sleep_between)
                break
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                time.sleep(3 * (attempt + 1))
                print(f"Retrying {area} (attempt {attempt + 2})…")

    raw = pd.concat(all_df, ignore_index=True)
    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])

    panel = (
        raw[["TIME_PERIOD", "REF_AREA_REQ", "OBS_VALUE"]]
        .rename(columns={"REF_AREA_REQ": "REF_AREA", "OBS_VALUE": "IR10Y"})
        .sort_values(["REF_AREA", "TIME_PERIOD"])
        .reset_index(drop=True)
    )

    wide = (
        panel.pivot(index="TIME_PERIOD", columns="REF_AREA", values="IR10Y")
        .sort_index()
    )

    print("\nDecember year-end preview:")
    dec = wide[wide.index.month == 12]
    print(dec.round(3).to_string())
    print("\nNote: DEU column is sourced from EA19 (euro-area 10Y yield).")

    return panel, wide


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    panel, wide = pull_ir10y()

    output_dir = Path(DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_path = output_dir / IR10Y_PANEL_FILE
    wide_path  = output_dir / IR10Y_WIDE_FILE

    panel.to_parquet(panel_path, index=False)
    wide.reset_index().to_parquet(wide_path, index=False)

    print(f"\nSaved:\n  {panel_path}\n  {wide_path}")
