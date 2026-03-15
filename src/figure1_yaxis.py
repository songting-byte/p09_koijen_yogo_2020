"""pull_st_rate.py
=================
Pull 3-month interbank rates (IR3TIB) from the OECD SDMX API
for the countries used in Figure 1.

Flow: OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0

Output (written to DATA_DIR):
  ST_RATE_PANEL_FILE  long-format CSV  (TIME_PERIOD, REF_AREA, IR3M)
  ST_RATE_WIDE_FILE   wide-format CSV  (TIME_PERIOD × country)

Configuration (via settings.py / .env / CLI):
  DATA_DIR              output directory       (default: _data/)
  OECD_START_PERIOD     start year             (default: "2003")
  OECD_END_PERIOD       end year               (default: "2020")
  ST_RATE_REF_AREAS     comma-separated list   (default: see below)
  ST_RATE_PANEL_FILE    output filename        (default: oecd_ir3tib_panel.csv)
  ST_RATE_WIDE_FILE     output filename        (default: oecd_ir3tib_wide.csv)

Dependencies: pip install requests pandas python-decouple
"""

import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from settings import config

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR     = config("DATA_DIR")
START_YEAR   = config("OECD_START_PERIOD", default="2003", cast=str)
END_YEAR     = config("OECD_END_PERIOD",   default="2020", cast=str)

# Monthly periods derived from annual settings
START = f"{START_YEAR}-01"
END   = f"{END_YEAR}-12"

_DEFAULT_REF_AREAS = "USA,EA19,JPN,CHE,GBR"
ST_RATE_REF_AREAS  = config("ST_RATE_REF_AREAS", default=_DEFAULT_REF_AREAS, cast=str)
REF_AREAS          = [c.strip() for c in ST_RATE_REF_AREAS.split(",")]

ST_RATE_PANEL_FILE = config("ST_RATE_PANEL_FILE", default="oecd_ir3tib_panel.parquet", cast=str)
ST_RATE_WIDE_FILE  = config("ST_RATE_WIDE_FILE",  default="oecd_ir3tib_wide.parquet",  cast=str)

# ── OECD endpoint ─────────────────────────────────────────────────────────────

FLOW_REF = "OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0"
BASE_URL  = f"https://sdmx.oecd.org/public/rest/data/{FLOW_REF}"

# ── Fetch helpers ─────────────────────────────────────────────────────────────

def fetch_one(ref_area: str) -> pd.DataFrame:
    """Fetch IR3TIB monthly series for a single country/region."""
    key = f"{ref_area}.M.IR3TIB.PA....."
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


def pull_st_rates(
    ref_areas: list[str] = REF_AREAS,
    max_retries: int = 3,
    sleep_between: float = 1.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull IR3TIB for all requested areas and return (panel, wide) DataFrames."""
    all_df = []
    for area in ref_areas:
        for attempt in range(max_retries):
            try:
                df = fetch_one(area)
                df["REF_AREA_REQ"] = area
                all_df.append(df)
                print(f"OK: {area}, rows={len(df)}")
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
        raw[["TIME_PERIOD", "REF_AREA", "OBS_VALUE"]]
        .rename(columns={"OBS_VALUE": "IR3M"})
        .sort_values(["REF_AREA", "TIME_PERIOD"])
        .reset_index(drop=True)
    )

    wide = (
        panel.pivot(index="TIME_PERIOD", columns="REF_AREA", values="IR3M")
        .sort_index()
    )

    return panel, wide


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    panel, wide = pull_st_rates()

    output_dir = Path(DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_path = output_dir / ST_RATE_PANEL_FILE
    wide_path  = output_dir / ST_RATE_WIDE_FILE

    panel.to_parquet(panel_path, index=False)
    wide.reset_index().to_parquet(wide_path, index=False)

    print(f"\nSaved:\n  {panel_path}\n  {wide_path}")
    # print(panel.head())
    # print(wide.tail())